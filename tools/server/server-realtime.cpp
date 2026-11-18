#include "server-realtime.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"
#include "server-turn.h"
#include "server-voice.h"
#include "server-asr.h"

#include "base64.hpp"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

//
// a realtime conversation, modeled after the OpenAI realtime API (GA)
//
// one slot holds the conversation for the whole session: text and audio go into it as they arrive, so a response
// starts right after the last audio. session.type "realtime" holds a conversation, "transcription" transcribes each
// committed audio buffer on its own.
//
// cascade: a model without audio input hears through the ASR model (--asr-model), the transcript of each user audio
// item goes into the conversation as its text; a voice model (--voice-model) speaks the responses.
//

namespace {

// output_modalities of a response, unset: the model switches between text and audio itself
bool parse_output_modalities(const json & j, mtmd_helper_gen_audio_out_mode & mode, std::string & err) {
    if (j.is_null()) {
        mode = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
        return true;
    }
    if (!j.is_array() || j.empty()) {
        err = "output_modalities must be a non-empty array";
        return false;
    }
    bool text  = false;
    bool audio = false;
    for (const auto & m : j) {
        if (m == "text") {
            text = true;
        } else if (m == "audio") {
            audio = true;
        } else {
            err = "output_modalities: unknown modality " + m.dump();
            return false;
        }
    }
    if (text && audio) {
        mode = MTMD_HELPER_GEN_AUDIO_OUT_MIXED;
    } else {
        mode = audio ? MTMD_HELPER_GEN_AUDIO_OUT_AUDIO : MTMD_HELPER_GEN_AUDIO_OUT_TEXT;
    }
    return true;
}

bool parse_max_output_tokens(const json & j, int32_t & n) {
    if (j == "inf") {
        n = -1;
        return true;
    }
    if (!j.is_number_integer() || j.get<int64_t>() < 1) {
        return false;
    }
    n = (int32_t) std::min<int64_t>(j.get<int64_t>(), INT32_MAX);
    return true;
}

// realtime tools have the format of the Responses API
bool parse_tools(const json & j, std::vector<common_chat_tool> & tools, std::string & err) {
    try {
        auto parsed = common_chat_tools_parse_oaicompat(server_chat_convert_responses_tools(j));
        // reject the schemas that would fail later, when the chat template builds the tool call grammar
        for (const auto & tool : parsed) {
            json_schema_to_grammar(json::parse(tool.parameters));
        }
        tools = std::move(parsed);
    } catch (const std::exception & e) {
        err = std::string("tools: ") + e.what();
        return false;
    }
    return true;
}

bool parse_tool_choice(const json & j, common_chat_tool_choice & choice, std::string & err) {
    // {"type": "function", "name": ...}: a tool call is required, the model picks among the tools
    if (j.is_object() && json_value(j, "type", std::string()) == "function") {
        choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        return true;
    }
    if (!j.is_string()) {
        err = "tool_choice must be \"auto\", \"none\", \"required\" or a function";
        return false;
    }
    try {
        choice = common_chat_tool_choice_parse_oaicompat(j.get<std::string>());
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
    return true;
}

const char * tool_choice_str(common_chat_tool_choice choice) {
    switch (choice) {
        case COMMON_CHAT_TOOL_CHOICE_NONE:     return "none";
        case COMMON_CHAT_TOOL_CHOICE_REQUIRED: return "required";
        default:                               return "auto";
    }
}

json output_modalities_json(mtmd_helper_gen_audio_out_mode mode) {
    switch (mode) {
        case MTMD_HELPER_GEN_AUDIO_OUT_TEXT:  return json::array({"text"});
        case MTMD_HELPER_GEN_AUDIO_OUT_AUDIO: return json::array({"audio"});
        case MTMD_HELPER_GEN_AUDIO_OUT_MIXED: return json::array({"text", "audio"});
        default:                 return nullptr;
    }
}

json max_output_tokens_json(int32_t n) {
    return n < 0 ? json("inf") : json(n);
}

// turn_detection: null, server_vad or semantic_vad; fields left out keep their defaults
bool parse_turn_detection(const json & j, std::optional<server_turn_params> & out, std::string & err) {
    if (j.is_null()) {
        out.reset();
        return true;
    }
    if (!j.is_object()) {
        err = "turn_detection must be an object or null";
        return false;
    }
    server_turn_params p;
    p.type = json_value(j, "type", std::string("server_vad"));
    if (p.type != "server_vad" && p.type != "semantic_vad") {
        err = "turn_detection: unknown type " + p.type;
        return false;
    }
    p.create_response    = json_value(j, "create_response", p.create_response);
    p.interrupt_response = json_value(j, "interrupt_response", p.interrupt_response);
    if (p.type == "server_vad") {
        p.threshold           = json_value(j, "threshold", p.threshold);
        p.prefix_padding_ms   = json_value(j, "prefix_padding_ms", p.prefix_padding_ms);
        p.silence_duration_ms = json_value(j, "silence_duration_ms", p.silence_duration_ms);
        if (j.contains("idle_timeout_ms") && !j["idle_timeout_ms"].is_null()) {
            p.idle_timeout_ms = j["idle_timeout_ms"].get<int>();
        }
        if (p.threshold <= 0.0f || p.threshold >= 1.0f || p.prefix_padding_ms < 0 || p.silence_duration_ms < 0) {
            err = "turn_detection: threshold must be in (0, 1), prefix_padding_ms and silence_duration_ms >= 0";
            return false;
        }
    } else {
        p.eagerness = json_value(j, "eagerness", p.eagerness);
        if (p.eagerness != "low" && p.eagerness != "medium" && p.eagerness != "high" && p.eagerness != "auto") {
            err = "turn_detection: eagerness must be low, medium, high or auto";
            return false;
        }
    }
    out = p;
    return true;
}

json turn_detection_json(const std::optional<server_turn_params> & p) {
    if (!p) {
        return nullptr;
    }
    if (p->type == "semantic_vad") {
        return {{"type", p->type}, {"eagerness", p->eagerness}, {"create_response", p->create_response},
                {"interrupt_response", p->interrupt_response}};
    }
    return {{"type", p->type}, {"threshold", p->threshold}, {"prefix_padding_ms", p->prefix_padding_ms},
            {"silence_duration_ms", p->silence_duration_ms},
            {"idle_timeout_ms", p->idle_timeout_ms < 0 ? json(nullptr) : json(p->idle_timeout_ms)},
            {"create_response", p->create_response}, {"interrupt_response", p->interrupt_response}};
}

// truncation: the oldest items leave the context of the model when the conversation does not fit in it
struct truncation_params {
    bool    on    = true;
    double  ratio = 0.5; // share of the limit the conversation keeps then, so that it does not happen at every turn
    int32_t limit = 0;   // token_limits.post_instructions, 0: the context
};

// "auto", "disabled" or {"type": "retention_ratio", "retention_ratio": 0.8, "token_limits": {"post_instructions": 4000}}
bool parse_truncation(const json & j, truncation_params & out, std::string & err) {
    truncation_params p;
    if (j == "disabled") {
        p.on = false;
    } else if (j.is_object() && json_value(j, "type", std::string()) == "retention_ratio") {
        if (!j.contains("retention_ratio") || !j["retention_ratio"].is_number()) {
            err = "truncation: retention_ratio is missing";
            return false;
        }
        p.ratio = j["retention_ratio"].get<double>();
        if (p.ratio < 0.0 || p.ratio > 1.0) {
            err = "truncation: retention_ratio must be between 0 and 1";
            return false;
        }
        const json limits = j.value("token_limits", json::object());
        if (limits.is_object() && limits.contains("post_instructions") && !limits["post_instructions"].is_null()) {
            const json & n = limits["post_instructions"];
            if (!n.is_number_integer() || n.get<int64_t>() < 1) {
                err = "truncation: token_limits.post_instructions must be a positive integer";
                return false;
            }
            p.limit = (int32_t) std::min<int64_t>(n.get<int64_t>(), INT32_MAX);
        }
    } else if (j != "auto") {
        err = "truncation must be \"auto\", \"disabled\" or a retention_ratio object";
        return false;
    }
    out = p;
    return true;
}

// a field of an object, an empty object if it is not there
json child(const json & j, const char * key) {
    return j.is_object() && j.contains(key) ? j[key] : json::object();
}

//
// audio formats of the client: PCM16 at any rate (24 kHz by default), G.711 u-law and A-law at 8 kHz
//

struct audio_format {
    std::string type = "audio/pcm";
    int         rate = 24000;

    json to_json() const {
        return type == "audio/pcm" ? json{{"type", type}, {"rate", rate}} : json{{"type", type}};
    }
};

bool parse_audio_format(const json & j, audio_format & fmt, std::string & err) {
    if (!j.is_object()) {
        err = "format must be an object";
        return false;
    }
    const std::string type = json_value(j, "type", std::string("audio/pcm"));
    if (type == "audio/pcm") {
        const int rate = json_value(j, "rate", 24000);
        if (rate < 8000 || rate > 48000) {
            err = "format: audio/pcm rate must be between 8000 and 48000";
            return false;
        }
        fmt = {type, rate};
    } else if (type == "audio/pcmu" || type == "audio/pcma") {
        fmt = {type, 8000};
    } else {
        err = "format: unknown type " + type + ", expected audio/pcm, audio/pcmu or audio/pcma";
        return false;
    }
    return true;
}

float ulaw_to_float(uint8_t u) {
    u = ~u;
    const int t = (((u & 0x0f) << 3) + 0x84) << ((u & 0x70) >> 4);
    return ((u & 0x80) ? 0x84 - t : t - 0x84) / 32768.0f;
}

float alaw_to_float(uint8_t a) {
    a ^= 0x55;
    const int seg = (a & 0x70) >> 4;
    int t = (a & 0x0f) << 4;
    t = seg == 0 ? t + 8 : (t + 0x108) << (seg - 1);
    return ((a & 0x80) ? t : -t) / 32768.0f;
}

uint8_t float_to_ulaw(float x) {
    int v = (int) std::lround(std::max(-1.0f, std::min(1.0f, x)) * 32767.0f);
    const uint8_t sign = v < 0 ? 0x80 : 0;
    v = std::min(std::abs(v), 32635) + 0x84;
    int seg = 7;
    for (int mask = 0x4000; seg > 0 && !(v & mask); mask >>= 1) {
        seg--;
    }
    return ~(sign | (seg << 4) | ((v >> (seg + 3)) & 0x0f));
}

uint8_t float_to_alaw(float x) {
    int v = (int) std::lround(std::max(-1.0f, std::min(1.0f, x)) * 32767.0f);
    const uint8_t sign = v >= 0 ? 0x80 : 0;
    v = std::min(std::abs(v), 32767) >> 3;
    uint8_t out;
    if (v < 32) {
        out = v >> 1;
    } else {
        int seg = 1;
        while (seg < 7 && v >= (64 << (seg - 1))) {
            seg++;
        }
        out = (seg << 4) | ((v >> seg) & 0x0f);
    }
    return (out | sign) ^ 0x55;
}

std::vector<float> decode_audio(const std::string & bytes, const audio_format & fmt) {
    std::vector<float> out;
    if (fmt.type == "audio/pcm") {
        out.resize(bytes.size() / 2);
        for (size_t i = 0; i < out.size(); i++) {
            int16_t v;
            memcpy(&v, bytes.data() + 2 * i, sizeof(v));
            out[i] = v / 32768.0f;
        }
    } else {
        const bool ulaw = fmt.type == "audio/pcmu";
        out.resize(bytes.size());
        for (size_t i = 0; i < bytes.size(); i++) {
            out[i] = ulaw ? ulaw_to_float((uint8_t) bytes[i]) : alaw_to_float((uint8_t) bytes[i]);
        }
    }
    return out;
}

std::string encode_audio(const std::vector<float> & samples, const audio_format & fmt) {
    std::string bytes;
    if (fmt.type == "audio/pcm") {
        bytes.resize(samples.size() * sizeof(int16_t));
        for (size_t i = 0; i < samples.size(); i++) {
            const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, samples[i])) * 32767.0f);
            memcpy(&bytes[2 * i], &v, sizeof(v));
        }
    } else {
        const bool ulaw = fmt.type == "audio/pcmu";
        bytes.resize(samples.size());
        for (size_t i = 0; i < samples.size(); i++) {
            bytes[i] = (char) (ulaw ? float_to_ulaw(samples[i]) : float_to_alaw(samples[i]));
        }
    }
    return bytes;
}

// streaming resampler: windowed sinc, low-pass at the lower of the two rates
struct pcm_resampler {
    int    in_rate  = 0;
    int    out_rate = 0;
    int    half     = 0;   // kernel half width, in input samples
    double cutoff   = 1.0; // relative to the input Nyquist
    double pos      = 0;   // position of the next output sample in buf
    std::vector<float> buf;

    void reset(int in, int out) {
        in_rate  = in;
        out_rate = out;
        cutoff   = std::min(1.0, (double) out / in);
        half     = (int) std::ceil(16 / cutoff);
        pos      = half;
        buf.assign(half, 0.0f); // zero history before the first sample
    }

    std::vector<float> process(const float * x, size_t n) {
        if (in_rate == out_rate) {
            return std::vector<float>(x, x + n);
        }
        buf.insert(buf.end(), x, x + n);
        std::vector<float> out;
        const double step = (double) in_rate / out_rate;
        while (pos + half < (double) buf.size()) {
            const int c = (int) std::floor(pos);
            double acc = 0;
            for (int i = c - half + 1; i <= c + half; i++) {
                const double t = pos - i;
                const double w = 0.5 + 0.5 * std::cos(M_PI * t / half);
                const double a = M_PI * cutoff * t;
                acc += buf[i] * cutoff * (std::fabs(a) < 1e-9 ? 1.0 : std::sin(a) / a) * w;
            }
            out.push_back((float) acc);
            pos += step;
        }
        // keep the history the next samples need
        const int drop = std::max(0, (int) std::floor(pos) - half + 1);
        buf.erase(buf.begin(), buf.begin() + std::min<size_t>(drop, buf.size()));
        pos -= drop;
        return out;
    }

    // the samples still held back by the kernel, then back to the initial state
    std::vector<float> flush() {
        std::vector<float> out;
        if (in_rate != out_rate) {
            const std::vector<float> pad(half, 0.0f);
            out = process(pad.data(), pad.size());
        }
        reset(in_rate, out_rate);
        return out;
    }
};

struct realtime_conversation {
    const server_realtime_deps &  deps;
    server_http_context::ws_conn & conn;

    std::string id      = "sess_" + random_string();
    std::string conv_id = "conv_" + random_string();

    // session config
    std::string                    type = "realtime";
    std::string                    instructions;
    std::string                    language; // transcription
    std::string                    out_language; // of a model that speaks itself, empty: model default
    mtmd_helper_gen_audio_out_mode out_mode          = MTMD_HELPER_GEN_AUDIO_OUT_AUDIO; // text if the model cannot speak
    int32_t                        max_output_tokens = -1; // "inf"
    audio_format                   in_fmt;                 // audio of the client
    audio_format                   out_fmt;
    int                            model_in_rate     = 16000;
    int                            model_out_rate    = 24000;

    // session fields kept for the client, they do not change the output
    json        transcription   = nullptr;
    json        noise_reduction = nullptr;
    std::string voice           = "default";
    double      speed           = 1.0;
    json        tracing         = nullptr;
    json        include         = nullptr;
    json        prompt          = nullptr;

    json              truncation = "auto"; // as the client sent it
    truncation_params trunc;

    std::optional<server_turn_params>     turn_params;     // null: the client commits the audio
    std::unique_ptr<server_turn_detector> detector;        // owned by the reader thread
    std::string                           speech_item_id;  // the user item of the speech in progress
    std::atomic<bool>                     responding{false};
    std::atomic<bool>                     hot{false}; // a response is in progress, see set_hot()
    std::atomic<bool>                     cancel_sent{false};   // the response in progress was cancelled, it ends soon
    std::atomic<bool>                     cancel_by_turn{false}; // by the start of speech (turn detection)
    std::atomic<bool>                     response_ended{false}; // for the idle timeout of the detector

    pcm_resampler in_rs;  // client rate -> model rate, owned by the reader thread
    pcm_resampler out_rs; // model rate -> client rate, owned by the result loop

    // tools, in the system turn
    json                          tools_json  = json::array(); // as the client sent them
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice       tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                          tool_choice_json = "auto"; // as the client sent it

    // the text around a turn, from the chat template
    std::string first_header; // system prompt + first user header
    std::string user_header;
    std::string user_footer;
    std::string assistant_header;
    std::string assistant_footer;
    std::string tool_header;  // around a tool result
    std::string tool_footer;
    bool        has_tool_text = false; // the chat template renders tool results
    std::string assistant_open;        // a first user turn and the generation prompt, as render_assistant() renders them

    // around a system message in the conversation, by what goes before it; next: before what follows it
    struct system_text {
        std::string after_assistant; // after a response (or another system message), next_header goes first
        std::string in_user;         // closes the open user turn
        std::string after_tool;      // after tool results
        std::string first;           // at the start, with the session instructions and tools
        std::string footer;
        std::string next;            // before the next user turn or the generation prompt
        bool        ok       = false;
        bool        ok_first = false;
        bool        ok_tool  = false;
    } sys_text;

    // the turn text of the contexts the session had, by context_key(): to derive it the chat template renders many times
    struct turn_text {
        std::string first_header, user_header, user_footer, assistant_header, assistant_footer, tool_header, tool_footer;
        std::string assistant_open;
        bool        has_tool_text;
        system_text sys;
        bool        ok;
    };
    static constexpr size_t          max_turn_texts = 16;
    std::map<std::string, turn_text> turn_texts;

    // conversation state, owned by the reader thread
    bool        started   = false; // something went into the slot, the session type is fixed
    bool        turn_open = false; // a user turn is being filled
    bool        has_audio = false; // the open turn has audio not committed yet
    bool        tool_results = false; // tool results went in after the last response, the next one answers them
    bool        system_last  = false; // a system message went in last, a user turn or a response follows it
    int         n_turns   = 0;
    std::string next_header;       // text that goes in before the next user turn

    // the turn text state, kept with each item: the conversation can be rebuilt from any item
    struct turn_state {
        bool        turn_open    = false;
        bool        tool_results = false;
        bool        system_last  = false;
        int         n_turns      = 0;
        std::string next_header;
    };

    // an item of the conversation and what it takes to put it in again
    struct conv_item {
        enum kind_t { USER_TEXT, USER_AUDIO, TOOL_OUTPUT, ASSISTANT, SYSTEM } kind;
        std::string          id;
        std::string          mark;        // slot mark at its start; empty: it follows the previous item of its response
        turn_state           before;
        json                 item;        // as the client sees it
        std::string          text;        // user text, tool output
        std::vector<float>   pcm;         // user audio at the model rate
        std::string          transcript;  // cascade: the text of the user audio, what the conversation holds
        std::string          response_id; // assistant items of one response go in together
        std::vector<std::pair<int64_t, int64_t>> spoken; // assistant audio of a voice: (end sample, end of its text) per piece
        int64_t              audio_samples = 0; // assistant audio sent, at audio_rate
        int                  audio_rate    = 0;
        std::string          audio;          // the audio as the client sent or got it (retrieve)
        audio_format         audio_fmt;
    };
    std::vector<conv_item> items;     // guarded by mutex
    std::vector<conv_item> old_items; // before them: out of the context by truncation, the client still has them (guarded by mutex)
    int                    n_marks = 0;

    // the conversation state belongs to the thread that holds conv_mutex: the reader thread for a client event, the
    // result loop to put the conversation in again after a slot error (recover); taken before mutex
    std::mutex        conv_mutex;
    std::atomic<bool> fault_pending{false}; // the result loop waits for conv_mutex
    std::atomic<int>  n_faults{0};          // slot errors other than a full context since the last good response
    int               n_clears = 0;         // CLEAR controls sent: a slot error from before the last one is void
    bool              ctx_lost = false;     // the conversation is not in the slot, the next response puts it in again

    // the response in progress, to start it again after a slot error
    json resp_event;
    bool resp_from_client = true;
    int  resp_retries     = 0;
    bool resp_tight       = false; // without the room kept for the answer: the turn to answer is all the context holds

    // the item being filled: its mark and turn state, set before its first content
    std::string pending_mark;
    turn_state  pending_before;
    bool        audio_open = false;    // the input audio buffer has an item mark
    int64_t     buf_samples = 0;       // audio the client appended to the buffer, at its rate
    int32_t     asr_in = 0, asr_out = 0; // cascade: tokens of the last transcription
    std::string client_event_id;       // event_id of the client event being handled, for its errors
    std::vector<float> pending_pcm;    // audio of the uncommitted buffer, at the model rate
    std::shared_ptr<server_asr_turn> asr_turn; // cascade: the transcription of the uncommitted buffer
    std::string resp_mark;             // mark and turn state at the start of the response in progress
    turn_state  resp_before;
    bool        resp_oob = false;      // conversation "none": the response stays out of the conversation (guarded by mutex)

    // a response with its own context (input, instructions or tools of the response): the conversation is put back
    // when it is done, client events wait until then
    bool                             pending_restore = false;
    std::string                      restore_mark;
    turn_state                       restore_before;
    std::string                      saved_instructions;
    std::vector<common_chat_tool>    saved_tools;
    json                             saved_tools_json;
    struct held_event {
        std::string                     msg;     // a client event
        std::vector<server_turn_action> actions; // or the turn detection of audio that came in meanwhile
    };
    std::deque<held_event>           held_events;

    // the items from here on are not in the slot (SIZE_MAX: all are): a delete takes its item out of the conversation
    // at once, the items after it go in again when the slot is needed, once for a run of deletes
    size_t stale_from  = SIZE_MAX;
    bool   last_append = false; // the last client event was audio

    // prompt cache of the contexts: a response with its own context starts from the snapshot of an earlier context
    // with the same instructions and tools whose items begin the conversation, and only its newer items go in; the
    // session context waits in a snapshot meanwhile, and only the items that came since go in when it is back
    struct ctx_snapshot {
        std::string              name;     // in the slot
        std::string              key;      // instructions and tools
        std::vector<std::string> ids;      // the items it holds, from the first one
        turn_state               turn;     // after them
        uint64_t                 used = 0;
    };
    static constexpr size_t   max_snapshots = 3;
    std::vector<ctx_snapshot> snapshots;        // own contexts of earlier responses
    bool                      snap_restore = false; // the session context of the response in progress is a snapshot
    ctx_snapshot              session_snap;
    ctx_snapshot              own_ctx;          // the own context in the slot, kept when the session context is back
    std::string               own_end_mark;     // after the items of the own context
    uint64_t                  n_snap_use = 0;
    int                       n_snaps    = 0;

    // the item or response the results belong to, shared with the result loop
    std::mutex  mutex;
    std::string last_item_id;
    std::string transcribe_item_id; // transcription: the item being transcribed
    std::string transcript;
    std::string response_id;
    std::string response_item_id;
    std::string response_text;
    std::string response_transcript;
    json        response_meta = json::object(); // fields of the response object besides id, status, output and usage
    bool        transcript_mode = false;        // the text of the response is the transcript of its audio
    audio_format resp_fmt;                      // audio format of the response in progress
    size_t      resp_item_pos   = 0;            // where its items go in the conversation
    int64_t     out_samples     = 0;            // audio of the response sent so far, at the output rate
    std::string out_audio;                      // and its bytes, for retrieve

    // a response spoken by a voice model: its text goes to the voice in pieces, the response is done after the
    // audio of the last piece (guarded by mutex, the voice callbacks run on the worker thread of the model)
    struct voice_segment {
        int64_t s0 = -1, s1 = -1; // output samples
        size_t  t0 = 0, t1 = 0;   // characters of the response text
    };
    bool                                    voice_resp    = false;
    int                                     voice_model   = 0;
    std::string                             voice_name;
    server_voice_segmenter                  segmenter;
    size_t                                  voice_text    = 0; // response text given to the segmenter
    int                                     voice_pending = 0;
    int32_t                                 voice_frames  = 0;
    bool                                    llm_done      = false;
    std::optional<server_task_result_realtime> held_done;
    std::shared_ptr<std::atomic<bool>>      voice_cancel;
    std::vector<voice_segment>              voice_segments;
    size_t                                  voice_sent    = 0; // transcript sent, it goes out with the audio of each piece
    int                                     voice_jobs    = 0; // jobs of the session not done yet, of any response
    int         idx_text  = -1;
    int         idx_audio = -1;
    json        parts     = json::array();
    json        output    = json::array(); // items of the response
    int         idx_message = -1;          // output index of the message, added with its first delta
    std::map<size_t, int> call_items;      // tool call -> output index
    std::optional<task_result_state> parse; // the tool call parser, with tools

    std::atomic<bool> closed{false};

    std::vector<std::thread> transcribers; // transcription of user audio, joined at the end of the session

    realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn)
        : deps(deps), conn(conn), out_language(deps.params.tts_lang) {
        if (speaks_itself() && !deps.params.tts_voice.empty()) {
            voice = deps.params.tts_voice;
        }
        // as the OpenAI API: a new session detects the turns and answers them, with speech if the model can speak
        if (!can_speak()) {
            out_mode = MTMD_HELPER_GEN_AUDIO_OUT_TEXT;
        }
        turn_params = server_turn_params();
        turn_params->silence_duration_ms = 200;
    }

    //
    // events
    //

    void send_event(json ev) {
        ev["event_id"] = "event_" + random_string();
        conn.send(safe_json_to_str(ev));
    }

    // event_id: of the client event that caused the error
    void send_error_event(const std::string & message, const std::string & type_err, const std::string & code, const std::string & event_id,
                          const std::string & param = "") {
        send_event({
            {"type",  "error"},
            {"error", {{"type", type_err}, {"code", code.empty() ? json(nullptr) : json(code)}, {"message", message},
                       {"param", param.empty() ? json(nullptr) : json(param)},
                       {"event_id", event_id.empty() ? json(nullptr) : json(event_id)}}},
        });
    }

    // an error of the client event being handled
    void send_error(const std::string & message, const std::string & type_err = "invalid_request_error", const std::string & code = "",
                    const std::string & param = "") {
        send_error_event(message, type_err, code, client_event_id, param);
    }

    // as the OpenAI API: a field the API does not have is an error; path: of the object in the client event
    bool known_fields(const json & j, const std::string & path, const std::vector<const char *> & names) {
        if (!j.is_object()) {
            return true;
        }
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (std::none_of(names.begin(), names.end(), [&](const char * n) { return it.key() == n; })) {
                const std::string param = path.empty() ? it.key() : path + "." + it.key();
                send_error("Unknown parameter: '" + param + "'.", "invalid_request_error", "unknown_parameter", param);
                return false;
            }
        }
        return true;
    }

    bool tools_fields(const json & list, const std::string & path) {
        for (size_t i = 0; list.is_array() && i < list.size(); i++) {
            if (json_value(list[i], "type", std::string("function")) == "function" &&
                !known_fields(list[i], path + "[" + std::to_string(i) + "]", {"type", "name", "description", "parameters"})) {
                return false;
            }
        }
        return true;
    }

    // the fields of a session object; audio.output.language is not in the OpenAI API (a model that speaks itself)
    bool session_fields(const json & s, bool realtime) {
        using names = std::vector<const char *>;
        const json audio = child(s, "audio");
        const json in    = child(audio, "input");
        const json out   = child(audio, "output");
        const json turn  = child(in, "turn_detection");
        const json trunc = child(s, "truncation");
        const bool semantic = json_value(turn, "type", std::string()) == "semantic_vad";
        return known_fields(s, "session", realtime ? names{"type", "audio", "include", "instructions", "max_output_tokens", "model",
                                                           "output_modalities", "parallel_tool_calls", "prompt", "reasoning", "tool_choice",
                                                           "tools", "tracing", "truncation"} :
                                                     names{"type", "audio", "include"}) &&
               known_fields(audio, "session.audio", realtime ? names{"input", "output"} : names{"input"}) &&
               known_fields(in, "session.audio.input", {"format", "noise_reduction", "transcription", "turn_detection"}) &&
               known_fields(child(in, "format"), "session.audio.input.format", {"type", "rate"}) &&
               known_fields(child(in, "noise_reduction"), "session.audio.input.noise_reduction", {"type", "strength"}) &&
               known_fields(child(in, "transcription"), "session.audio.input.transcription",
                            {"delay", "keywords", "language", "languages", "model", "prompt"}) &&
               known_fields(turn, "session.audio.input.turn_detection",
                            semantic ? names{"type", "create_response", "eagerness", "interrupt_response"} :
                                       names{"type", "create_response", "idle_timeout_ms", "interrupt_response", "prefix_padding_ms",
                                             "silence_duration_ms", "threshold"}) &&
               known_fields(out, "session.audio.output", {"format", "speed", "voice", "language"}) &&
               known_fields(child(out, "format"), "session.audio.output.format", {"type", "rate"}) &&
               known_fields(child(out, "voice"), "session.audio.output.voice", {"id"}) &&
               known_fields(trunc, "session.truncation", {"type", "retention_ratio", "token_limits"}) &&
               known_fields(child(trunc, "token_limits"), "session.truncation.token_limits", {"post_instructions"}) &&
               known_fields(child(s, "tool_choice"), "session.tool_choice", {"type", "name", "server_label"}) &&
               tools_fields(child(s, "tools"), "session.tools");
    }

    bool response_fields(const json & r) {
        const json audio = child(r, "audio");
        const json out   = child(audio, "output");
        return known_fields(r, "response", {"audio", "conversation", "input", "instructions", "max_output_tokens", "metadata",
                                            "output_modalities", "parallel_tool_calls", "prompt", "reasoning", "tool_choice", "tools"}) &&
               known_fields(audio, "response.audio", {"output"}) &&
               known_fields(out, "response.audio.output", {"format", "voice"}) &&
               known_fields(child(out, "format"), "response.audio.output.format", {"type", "rate"}) &&
               known_fields(child(out, "voice"), "response.audio.output.voice", {"id"}) &&
               known_fields(child(r, "tool_choice"), "response.tool_choice", {"type", "name", "server_label"}) &&
               tools_fields(child(r, "tools"), "response.tools");
    }

    // audio input through the ASR model: the model does not hear itself
    bool cascade() const {
        return deps.asr && !deps.cap.audio;
    }

    // language of the transcription: of audio.input.transcription, else of a transcription session
    std::string asr_language() const {
        const std::string l = transcription.is_object() ? json_value(transcription, "language", std::string()) : std::string();
        return l.empty() ? language : l;
    }

    // audio output: from a voice model if one is configured, else from a model that speaks itself
    bool can_speak() const {
        return deps.voices || deps.cap.audio_gen;
    }

    bool speaks_itself() const {
        return !deps.voices && deps.cap.audio_gen;
    }

    // voice of a model that speaks itself, empty: model default
    // a name the model does not have (e.g. an OpenAI voice) is kept for the client, the model speaks with its default
    std::string model_voice(const std::string & v) const {
        const auto & voices = deps.meta.tts_voices;
        const bool known = voices.empty() || std::find(voices.begin(), voices.end(), v) != voices.end();
        return v == "default" || !known ? std::string() : v;
    }

    // system prompt text for the output language and voice, for models that take them there (LFM3)
    std::string speech_prompt(const std::string & lang, const std::string & v) const {
        return common_chat_get_speech_system_prompt(deps.meta.chat_params.tmpls.get(), lang, model_voice(v));
    }

    json session_json() const {
        json audio_in = {
            {"format",          in_fmt.to_json()},
            {"transcription",   transcription},
            {"noise_reduction", noise_reduction},
            {"turn_detection",  turn_detection_json(turn_params)},
        };
        if (type == "transcription") {
            audio_in["transcription"] = {{"model", deps.meta.model_name}, {"language", language.empty() ? json(nullptr) : json(language)},
                                         {"prompt", nullptr}};
            return {
                {"id",         id},
                {"object",     "realtime.transcription_session"},
                {"type",       "transcription"},
                {"audio",      {{"input", audio_in}}},
                {"include",    include},
                {"expires_at", 0},
            };
        }
        json audio = {{"input", audio_in}};
        if (can_speak()) {
            audio["output"] = {{"format", out_fmt.to_json()}, {"voice", voice}, {"speed", speed}};
            if (!out_language.empty()) {
                audio["output"]["language"] = out_language;
            }
        }
        return {
            {"id",                id},
            {"object",            "realtime.session"},
            {"type",              "realtime"},
            {"model",             deps.meta.model_name},
            {"instructions",      instructions},
            {"output_modalities", output_modalities_json(out_mode)},
            {"max_output_tokens", max_output_tokens_json(max_output_tokens)},
            {"tools",             tools_json},
            {"tool_choice",       tool_choice_json},
            {"tracing",           tracing},
            {"truncation",        truncation},
            {"prompt",            prompt},
            {"include",           include},
            {"expires_at",        0}, // the session does not expire
            {"audio",             audio},
        };
    }

    //
    // slot controls, run on the slot in the order they are posted
    //

    bool control(server_control_action action, std::string text = {}, std::vector<float> audio = {},
                 server_control_response response = {}) {
        server_response_reader ctl(deps.queue_tasks, deps.queue_results, 1);
        server_task task(SERVER_TASK_TYPE_CONTROL);
        task.id                       = ctl.get_new_id();
        task.params.control_action    = action;
        task.params.control_cmpl_id   = id;
        task.params.control_text      = std::move(text);
        task.params.control_audio     = std::move(audio);
        task.params.control_response  = response;
        if (action == SERVER_CONTROL_ACTION_CLEAR) {
            n_clears++;
        }
        ctl.post_task(std::move(task));
        auto result = ctl.next([this]() { return closed.load() || !conn.is_open(); });
        auto * res = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
        return res && res->success;
    }

    // sampling of a response: the server defaults, with the tool call grammar of chat
    common_params_sampling response_sampling(const common_chat_session * chat) const {
        common_params_sampling sampling = deps.params.sampling;
        if (chat) {
            chat->apply_sampling(sampling);
        }
        return sampling;
    }

    //
    // chat template
    //

    common_chat_templates_inputs chat_inputs(const std::vector<common_chat_msg> & messages, bool add_generation_prompt) const {
        const auto & chat = deps.meta.chat_params;
        common_chat_templates_inputs inputs;
        inputs.messages              = messages;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja             = chat.use_jinja;
        inputs.enable_thinking       = false;
        inputs.reasoning_format      = chat.reasoning_format;
        inputs.chat_template_kwargs  = chat.chat_template_kwargs;
        inputs.tools                 = tools;
        return inputs;
    }

    std::string render(const std::vector<common_chat_msg> & messages, bool add_generation_prompt = false) const {
        return common_chat_templates_apply(deps.meta.chat_params.tmpls.get(), chat_inputs(messages, add_generation_prompt)).prompt.str();
    }

    std::string render_with_instructions(std::vector<common_chat_msg> messages, bool add_generation_prompt) const {
        if (!instructions.empty()) {
            common_chat_msg sys;
            sys.role    = "system";
            sys.content = instructions;
            messages.insert(messages.begin(), sys);
        }
        return render(messages, add_generation_prompt);
    }

    // an assistant turn as the chat template writes it in the history (header included); sets next_header to its footer
    std::string render_assistant(const std::string & content, const std::vector<common_chat_tool_call> & calls) {
        common_chat_msg user;
        user.role    = "user";
        user.content = "@@USER@@";
        common_chat_msg msg;
        msg.role       = "assistant";
        msg.content    = content;
        msg.tool_calls = calls;
        try {
            const std::string g = assistant_open.empty() ? render_with_instructions({user}, true) : assistant_open;
            const std::string c = render_with_instructions({user, msg}, false);
            if (c.compare(0, g.size(), g) == 0) {
                std::string body = c.substr(g.size());
                next_header.clear();
                if (body.size() >= assistant_footer.size() &&
                    body.compare(body.size() - assistant_footer.size(), assistant_footer.size(), assistant_footer) == 0) {
                    body.resize(body.size() - assistant_footer.size());
                    next_header = assistant_footer;
                }
                return assistant_header + body;
            }
        } catch (const std::exception &) {
            // the template rejects the message, the content goes in as it is
        }
        next_header = assistant_footer;
        return assistant_header + content;
    }

    static bool ends_with(const std::string & s, const std::string & suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    // the text around a user and an assistant turn, from placeholder messages
    bool derive_turn_text() {
        const std::string user  = "@@USER@@";
        const std::string reply = "@@REPLY@@";
        auto msg = [](const char * role, const std::string & content) {
            common_chat_msg m;
            m.role    = role;
            m.content = content;
            return m;
        };
        std::vector<common_chat_msg> sys;
        if (!instructions.empty()) {
            sys.push_back(msg("system", instructions));
        }
        auto with = [&](std::vector<common_chat_msg> rest, bool add_generation_prompt = false) {
            std::vector<common_chat_msg> all = sys;
            all.insert(all.end(), rest.begin(), rest.end());
            return render(all, add_generation_prompt);
        };
        const std::string a = with({msg("user", user)});
        const std::string g = with({msg("user", user)}, true);
        const std::string c = with({msg("user", user), msg("assistant", reply)});
        const std::string d = with({msg("user", user), msg("assistant", reply), msg("user", user)});

        const size_t pa = a.find(user);
        const size_t pc = c.find(reply);
        const size_t pd = d.find(user, c.size());
        if (pa == std::string::npos || pc == std::string::npos || pd == std::string::npos ||
            c.compare(0, a.size(), a) != 0 || d.compare(0, c.size(), c) != 0 || g.compare(0, a.size(), a) != 0 || pc < a.size()) {
            return false;
        }
        // the response opens with the generation prompt, a past turn may render differently (e.g. its think block)
        first_header     = a.substr(0, pa);
        user_footer      = a.substr(pa + user.size());
        assistant_header = g.substr(a.size());
        assistant_open   = g;
        assistant_footer = c.substr(pc + reply.size());
        user_header      = d.substr(c.size(), pd - c.size());

        // a system message in the conversation: its text after a response, inside an open user turn and at the start;
        // what follows it must be the user header or the generation prompt after a common part (sys_text.next)
        sys_text = {};
        const std::string sys_msg = "@@SYSTEM@@";
        const std::string user2   = "@@USER2@@";
        const auto        npos    = std::string::npos;
        try {
            const std::string f1 = with({msg("user", user), msg("assistant", reply), msg("system", sys_msg)});
            const std::string g1 = with({msg("user", user), msg("assistant", reply), msg("system", sys_msg)}, true);
            const std::string f2 = with({msg("user", user), msg("assistant", reply), msg("user", user2), msg("system", sys_msg)});
            const std::string f3 = with({msg("user", user), msg("assistant", reply), msg("system", sys_msg), msg("user", user2)});
            const size_t      p1 = f1.find(sys_msg, c.size());
            const size_t      q2 = f2.find(user2, c.size());
            const size_t      p2 = q2 == npos ? npos : f2.find(sys_msg, q2 + user2.size());
            const size_t      p3 = f3.find(user2, f1.size());
            if (p1 != npos && p2 != npos && p3 != npos && f1.compare(0, c.size(), c) == 0 && f2.compare(0, c.size(), c) == 0 &&
                f2.compare(c.size(), q2 - c.size(), user_header) == 0 && f3.compare(0, f1.size(), f1) == 0 &&
                g1.compare(0, f1.size(), f1) == 0) {
                const std::string footer = f1.substr(p1 + sys_msg.size());
                const std::string uh     = f3.substr(f1.size(), p3 - f1.size());
                const std::string gp     = g1.substr(f1.size());
                if (f2.substr(p2 + sys_msg.size()) == footer && ends_with(uh, user_header) && ends_with(gp, assistant_header) &&
                    uh.substr(0, uh.size() - user_header.size()) == gp.substr(0, gp.size() - assistant_header.size())) {
                    sys_text.after_assistant = f1.substr(c.size(), p1 - c.size());
                    sys_text.in_user         = f2.substr(q2 + user2.size(), p2 - q2 - user2.size());
                    sys_text.footer          = footer;
                    sys_text.next            = uh.substr(0, uh.size() - user_header.size());
                    sys_text.ok              = true;
                }
            }
            if (sys_text.ok) {
                const std::string f4 = with({msg("system", sys_msg)});
                const std::string f5 = with({msg("system", sys_msg), msg("user", user2)});
                const size_t      p4 = f4.find(sys_msg);
                const size_t      p5 = f5.find(user2, f4.size());
                if (p4 != npos && p5 != npos && f4.substr(p4 + sys_msg.size()) == sys_text.footer &&
                    f5.compare(0, f4.size(), f4) == 0 && f5.substr(f4.size(), p5 - f4.size()) == sys_text.next + user_header) {
                    sys_text.first    = f4.substr(0, p4);
                    sys_text.ok_first = true;
                }
            }
        } catch (const std::exception &) {
            // the template has no system messages in the conversation
        }

        // a tool result follows the assistant turn that called the tool; without session tools a placeholder tool
        // stands in, the tools can come with each response (response.create) and their results with the conversation
        tool_header.clear();
        tool_footer.clear();
        has_tool_text = false;
        const std::vector<common_chat_tool> session_tools = tools;
        if (tools.empty()) {
            tools.push_back({"tool", "", "{\"type\":\"object\",\"properties\":{}}"});
        }
        try {
            const std::string result = "@@RESULT@@";
            common_chat_msg call = msg("assistant", "");
            call.tool_calls.push_back({tools[0].name, "{}", "call_0"});
            common_chat_msg res = msg("tool", result);
            res.tool_call_id = "call_0";
            const std::string e  = with({msg("user", user), call});
            const std::string f  = with({msg("user", user), call, res});
            const size_t      pf = f.find(result, e.size());
            if (pf != std::string::npos && f.compare(0, e.size(), e) == 0) {
                tool_header   = f.substr(e.size(), pf - e.size());
                tool_footer   = f.substr(pf + result.size());
                has_tool_text = true;
            }
            // a system message after tool results
            if (has_tool_text && sys_text.ok) {
                const std::string f6 = with({msg("user", user), call, res, msg("system", sys_msg)});
                const size_t      pr = f6.find(result, e.size());
                if (pr != npos && f6.compare(0, e.size(), e) == 0 && f6.compare(e.size(), pr - e.size(), tool_header) == 0) {
                    const std::string rest = f6.substr(pr + result.size());
                    const size_t      ps   = rest.find(sys_msg);
                    if (ps != npos && rest.compare(0, tool_footer.size(), tool_footer) == 0 && ps >= tool_footer.size() &&
                        rest.substr(ps + sys_msg.size()) == sys_text.footer) {
                        sys_text.after_tool = rest.substr(tool_footer.size(), ps - tool_footer.size());
                        sys_text.ok_tool    = true;
                    }
                }
            }
        } catch (const std::exception &) {
            // the template has no tool messages
        }
        tools = session_tools;
        return has_tool_text || session_tools.empty();
    }

    turn_text save_turn_text(bool ok) const {
        return {first_header, user_header, user_footer, assistant_header, assistant_footer, tool_header, tool_footer,
                assistant_open, has_tool_text, sys_text, ok};
    }

    bool load_turn_text(const turn_text & t) {
        first_header     = t.first_header;
        user_header      = t.user_header;
        user_footer      = t.user_footer;
        assistant_header = t.assistant_header;
        assistant_footer = t.assistant_footer;
        tool_header      = t.tool_header;
        tool_footer      = t.tool_footer;
        assistant_open   = t.assistant_open;
        has_tool_text    = t.has_tool_text;
        sys_text         = t.sys;
        return t.ok;
    }

    //
    // conversation
    //

    // the header of a user turn goes in before its first audio or text
    bool ensure_turn() {
        if (turn_open) {
            return true;
        }
        const std::string header = n_turns == 0 && !system_last ? first_header : next_header + user_header;
        if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, header)) {
            return false;
        }
        next_header.clear();
        started     = true;
        turn_open   = true;
        system_last = false;
        return true;
    }

    // the text of a system message where the conversation stands, empty with err set if the template has none there
    std::string system_text_here(std::string & err) const {
        if (!sys_text.ok) {
            err = "the chat template has no system messages in the conversation";
            return "";
        }
        if (turn_open) {
            return sys_text.in_user;
        }
        if (tool_results) {
            if (!sys_text.ok_tool) {
                err = "the chat template has no system message after tool results";
                return "";
            }
            return sys_text.after_tool;
        }
        if (n_turns == 0 && !system_last) {
            if (!sys_text.ok_first) {
                err = "the chat template has no system message at the start of the conversation";
                return "";
            }
            return sys_text.first;
        }
        return next_header + sys_text.after_assistant;
    }

    // nothing is in the conversation yet
    bool at_start() const {
        return n_turns == 0 && !turn_open && !tool_results && !system_last;
    }

    turn_state save_turn() const {
        return {turn_open, tool_results, system_last, n_turns, next_header};
    }

    void load_turn(const turn_state & t) {
        turn_open    = t.turn_open;
        tool_results = t.tool_results;
        system_last  = t.system_last;
        n_turns      = t.n_turns;
        next_header  = t.next_header;
    }

    // marks the slot before the first content of an item
    bool begin_item() {
        pending_mark   = "m" + std::to_string(n_marks++);
        pending_before = save_turn();
        return control(SERVER_CONTROL_ACTION_MARK, pending_mark);
    }

    // a finished item: the client sees it, the conversation keeps it
    void add_item(conv_item it, bool notify = true) {
        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = it.id;
        if (notify) {
            send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", it.item}});
            send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", it.item}});
        }
        items.push_back(std::move(it));
    }

    // puts an item into the slot (as it arrives or when the conversation is rebuilt), from its own mark
    bool put_item(conv_item & it) {
        if (!begin_item()) {
            return false;
        }
        it.mark   = pending_mark;
        it.before = pending_before;
        switch (it.kind) {
            case conv_item::USER_TEXT:
                return ensure_turn() && control(SERVER_CONTROL_ACTION_TEXT_APPEND, it.text);
            case conv_item::USER_AUDIO:
                if (cascade()) {
                    // the conversation holds what was said, the audio of a client item is transcribed first
                    if (it.transcript.empty() && !it.pcm.empty()) {
                        int32_t n_in = 0, n_out = 0;
                        if (!deps.asr->transcribe(it.pcm, asr_language(), it.transcript, n_in, n_out)) {
                            send_error("the transcription of the audio failed", "server_error");
                            return false;
                        }
                        if (!it.item.is_null()) {
                            it.item["content"][0]["transcript"] = it.transcript;
                        }
                    }
                    return ensure_turn() && control(SERVER_CONTROL_ACTION_TEXT_APPEND, it.transcript);
                }
                // the whole audio with its end: the slot has the encoded audio of the last turns
                return ensure_turn() && control(SERVER_CONTROL_ACTION_AUDIO_END, {}, it.pcm);
            case conv_item::TOOL_OUTPUT:
                if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, next_header + tool_header + it.text + tool_footer)) {
                    return false;
                }
                next_header.clear();
                tool_results = true;
                system_last  = false;
                return true;
            case conv_item::SYSTEM:
                {
                    std::string err;
                    const std::string head = system_text_here(err);
                    if (!err.empty() || !control(SERVER_CONTROL_ACTION_TEXT_APPEND, head + it.text + sys_text.footer)) {
                        return false;
                    }
                    next_header  = sys_text.next;
                    started      = true;
                    turn_open    = false;
                    tool_results = false;
                    system_last  = true;
                    return true;
                }
            case conv_item::ASSISTANT:
                break;
        }
        return false;
    }

    // the items of one response go in as one assistant turn, rendered by the chat template
    bool put_response(std::vector<conv_item> & group) {
        // at the start of the conversation the system prompt comes with the first user header: an empty user turn
        if (at_start() && !ensure_turn()) {
            return false;
        }
        if (!begin_item()) {
            return false;
        }
        std::string content;
        std::vector<common_chat_tool_call> calls;
        for (size_t i = 0; i < group.size(); i++) {
            group[i].mark   = i == 0 ? pending_mark : "";
            group[i].before = pending_before;
            const json & item = group[i].item;
            if (json_value(item, "type", std::string()) == "function_call") {
                calls.push_back({json_value(item, "name", std::string()), json_value(item, "arguments", std::string()),
                                 json_value(item, "call_id", std::string())});
                continue;
            }
            if (!group[i].text.empty()) {
                content += group[i].text;
                continue;
            }
            for (const auto & part : item.value("content", json::array())) {
                content += json_value(part, "text", json_value(part, "transcript", std::string()));
            }
        }
        const std::string text = (turn_open ? user_footer : tool_results ? "" : next_header) + render_assistant(content, calls);
        if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, text)) {
            return false;
        }
        turn_open    = false;
        tool_results = false;
        system_last  = false;
        n_turns++;
        return true;
    }

    void on_append(const json & ev) {
        std::string pcm;
        try {
            pcm = base64::decode(json_value(ev, "audio", std::string()));
        } catch (const std::exception &) {
            send_error("invalid base64 audio", "invalid_request_error", "invalid_value");
            return;
        }
        const std::vector<float> in = decode_audio(pcm, in_fmt);
        if (in.empty()) {
            return;
        }
        buf_samples += (int64_t) in.size();
        std::vector<float> samples = in_rs.process(in.data(), in.size());
        if (detector) {
            on_turn_audio(samples);
            return;
        }
        if (!open_audio()) {
            return;
        }
        has_audio = true; // samples held back by the resampler come with the next audio or the commit
        append_audio(std::move(samples));
    }

    // the input audio buffer starts a user audio item
    bool open_audio() {
        if (audio_open) {
            return true;
        }
        if (!put_stale() || !begin_item() || !ensure_turn()) {
            return false;
        }
        audio_open = true;
        pending_pcm.clear();
        if (cascade()) {
            asr_turn = deps.asr->open(asr_language());
        }
        return true;
    }

    bool append_audio(std::vector<float> samples) {
        if (samples.empty()) {
            return true;
        }
        if (!open_audio()) {
            return false;
        }
        pending_pcm.insert(pending_pcm.end(), samples.begin(), samples.end());
        if (asr_turn) {
            deps.asr->push(asr_turn, std::move(samples));
        } else if (!control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, std::move(samples))) {
            return false;
        }
        has_audio = true;
        return true;
    }

    // the turn model runs on 16 kHz audio, like Silero
    server_turn_model * turn_model_for_session() const {
        return deps.turn_model && model_in_rate == deps.turn_model->sample_rate() ? deps.turn_model.get() : nullptr;
    }

    // Silero runs on 16 kHz audio, at other model rates the level-based VAD is used
    std::unique_ptr<server_vad> make_vad() const {
        if (deps.vad_model && model_in_rate == 16000) {
            return server_vad_silero(deps.vad_model);
        }
        return server_vad_energy(model_in_rate);
    }

    // turn detection: the audio of a turn goes into the conversation from the start of speech, its end commits it
    void on_turn_audio(const std::vector<float> & samples) {
        if (response_ended.exchange(false)) {
            detector->response_done();
        }
        std::vector<server_turn_action> actions = detector->push(samples.data(), samples.size());
        if (!holding()) {
            run_turn_actions(actions);
            return;
        }
        // a response with its own context is in progress: the start of speech interrupts it now, the audio and the
        // commit wait with the other client events until the conversation is back
        held_event held;
        for (auto & a : actions) {
            if (a.kind == server_turn_action::SPEECH_STARTED && held.actions.empty()) {
                on_speech_started(a);
            } else {
                held.actions.push_back(std::move(a));
            }
        }
        if (!held.actions.empty()) {
            held_events.push_back(std::move(held));
        }
    }

    // client events wait while a response with its own context is in progress (see handle)
    bool holding() const {
        return pending_restore && responding;
    }

    void on_speech_started(const server_turn_action & a) {
        speech_item_id = "item_" + random_string();
        if (turn_params->interrupt_response && responding) {
            cancel_voice();
            cancel_by_turn = true;
            cancel_sent    = true;
            control(SERVER_CONTROL_ACTION_RESPONSE_CANCEL);
        }
        send_event({{"type", "input_audio_buffer.speech_started"}, {"audio_start_ms", a.audio_ms},
                    {"item_id", speech_item_id}});
    }

    void run_turn_actions(std::vector<server_turn_action> & actions) {
        for (auto & a : actions) {
            switch (a.kind) {
                case server_turn_action::SPEECH_STARTED:
                    {
                        on_speech_started(a);
                    } break;
                case server_turn_action::AUDIO:
                    {
                        append_audio(std::move(a.audio));
                    } break;
                case server_turn_action::SPEECH_STOPPED:
                    {
                        send_event({{"type", "input_audio_buffer.speech_stopped"}, {"audio_end_ms", a.audio_ms},
                                    {"item_id", speech_item_id}});
                        if (commit_turn() && type == "realtime" && turn_params->create_response) {
                            on_response_create(json::object(), false);
                        }
                    } break;
                case server_turn_action::IDLE_TIMEOUT:
                    {
                        // the user stayed silent: the model takes the turn
                        const std::string item = "item_" + random_string();
                        send_event({{"type", "input_audio_buffer.timeout_triggered"}, {"audio_start_ms", a.audio_ms},
                                    {"audio_end_ms", a.audio_ms}, {"item_id", item}});
                        if (type == "realtime" && turn_params->create_response && ensure_turn()) {
                            on_response_create(json::object(), false);
                        }
                    } break;
            }
        }
    }

    // ends the audio of the open turn, it becomes a user item
    bool commit() {
        if (!has_audio) {
            send_error("input_audio_buffer.commit: the buffer is empty", "invalid_request_error", "input_audio_buffer_commit_empty");
            return false;
        }
        if (!append_audio(in_rs.flush())) {
            return false;
        }
        // cascade: the turn has its text when the ASR model is done, the client events after the commit wait for it
        std::string asr_text;
        bool        asr_ok = true;
        asr_in  = 0;
        asr_out = 0;
        if (asr_turn) {
            std::promise<bool> done;
            deps.asr->finish(asr_turn, [&](bool ok, const std::string & text, int32_t n_in, int32_t n_out) {
                asr_text = text;
                asr_in   = n_in;
                asr_out  = n_out;
                done.set_value(ok);
            });
            asr_ok = done.get_future().get();
            asr_turn.reset();
            if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, asr_text)) {
                return false;
            }
        } else if (!control(SERVER_CONTROL_ACTION_AUDIO_END)) {
            return false;
        }
        has_audio   = false;
        audio_open  = false;
        buf_samples = 0;

        conv_item it;
        it.kind   = conv_item::USER_AUDIO;
        it.id     = speech_item_id.empty() ? "item_" + random_string() : speech_item_id;
        it.mark   = pending_mark;
        it.before = pending_before;
        it.pcm    = std::move(pending_pcm);
        {
            // the audio of the item for retrieve, in the client's format: what went into the conversation
            pcm_resampler rs;
            rs.reset(model_in_rate, in_fmt.rate);
            std::vector<float> x = rs.process(it.pcm.data(), it.pcm.size());
            const std::vector<float> t = rs.flush();
            x.insert(x.end(), t.begin(), t.end());
            it.audio     = encode_audio(x, in_fmt);
            it.audio_fmt = in_fmt;
        }
        it.item   = {{"id", it.id}, {"type", "message"}, {"status", "completed"}, {"role", "user"},
                     {"content", json::array({{{"type", "input_audio"}, {"transcript", nullptr}}})}};
        if (cascade()) {
            it.transcript = asr_text;
            it.item["content"][0]["transcript"] = asr_text;
        }
        pending_pcm.clear();
        speech_item_id.clear();
        {
            std::lock_guard<std::mutex> lock(mutex);
            const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
            send_event({{"type", "input_audio_buffer.committed"}, {"previous_item_id", prev}, {"item_id", it.id}});
        }
        const std::string id = it.id;
        std::vector<float> pcm = it.pcm;
        add_item(std::move(it));
        if (cascade()) {
            if (type == "realtime" && !transcription.is_null()) {
                std::lock_guard<std::mutex> lock(mutex);
                send_transcription(id, asr_ok, asr_text, asr_in, asr_out);
            }
        } else {
            transcribe_item(id, std::move(pcm));
        }
        return true;
    }

    // the transcription events of a user audio item, with the mutex held
    void send_transcription(const std::string & id, bool ok, const std::string & text, int32_t n_in, int32_t n_out) {
        if (!ok) {
            send_event({{"type", "conversation.item.input_audio_transcription.failed"}, {"item_id", id}, {"content_index", 0},
                        {"error", {{"type", "transcription_error"}, {"code", nullptr}, {"message", "transcription failed"}, {"param", nullptr}}}});
            return;
        }
        if (conv_item * it = any_item(id)) {
            it->item["content"][0]["transcript"] = text;
        }
        send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", id}, {"content_index", 0}, {"delta", text}});
        send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", id}, {"content_index", 0},
                    {"transcript", text}, {"usage", transcription_usage(n_in, n_out)}});
    }

    // the audio embeddings of the prompt are not counted apart, the text of the ASR prompt is small
    static json transcription_usage(int32_t n_in, int32_t n_out) {
        return {{"type", "tokens"}, {"total_tokens", n_in + n_out}, {"input_tokens", n_in}, {"output_tokens", n_out},
                {"input_token_details", {{"text_tokens", 0}, {"audio_tokens", n_in}}}};
    }

    // audio.input.transcription: the user audio is transcribed by the ASR model or on another slot, the conversation
    // does not wait for it
    void transcribe_item(const std::string & id, std::vector<float> pcm) {
        const bool by_asr = deps.asr && deps.asr->sample_rate() == model_in_rate;
        if (type != "realtime" || transcription.is_null() || (!by_asr && !deps.transcribe)) {
            return;
        }
        if (!by_asr && deps.params.n_parallel < 2) {
            // the session holds the only slot, the transcription would wait for it until the session ends
            std::lock_guard<std::mutex> lock(mutex);
            send_event({{"type", "conversation.item.input_audio_transcription.failed"}, {"item_id", id}, {"content_index", 0},
                        {"error", {{"type", "transcription_error"}, {"code", nullptr}, {"param", nullptr},
                                   {"message", "the transcription needs a second slot (--parallel) or an ASR model (--asr-model)"}}}});
            return;
        }
        const std::string lang = json_value(transcription, "language", std::string());
        transcribers.emplace_back([this, id, lang, by_asr, pcm = std::move(pcm)]() {
            std::string text;
            int32_t n_in = 0, n_out = 0;
            const bool ok = by_asr ? deps.asr->transcribe(pcm, lang, text, n_in, n_out) :
                                     deps.transcribe(pcm, model_in_rate, lang, text, n_in, n_out, [this]() { return closed.load(); });
            if (closed) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            send_transcription(id, ok, text, n_in, n_out);
        });
    }

    void on_commit() {
        // as OpenAI: a buffer the client commits has at least 100 ms of audio
        const int64_t ms = buf_samples * 1000 / in_fmt.rate;
        if (has_audio && !detector && ms < 100) {
            send_error("input_audio_buffer.commit: the buffer is too small, expected at least 100 ms of audio, it has " +
                       std::to_string(ms) + " ms", "invalid_request_error", "input_audio_buffer_commit_empty");
            return;
        }
        commit_turn();
    }

    // the end of a user turn: its audio becomes an item, in a transcription session its text follows
    bool commit_turn() {
        if (!commit()) {
            return false;
        }
        if (type != "transcription") {
            return true;
        }
        if (cascade()) {
            // the ASR model has transcribed the buffer already, the slot only holds the turn header
            std::string id, text;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!items.empty()) {
                    id   = items.back().id;
                    text = items.back().transcript;
                }
            }
            control(SERVER_CONTROL_ACTION_CLEAR);
            turn_open = false;
            n_turns   = 0;
            std::lock_guard<std::mutex> lock(mutex);
            send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", id}, {"content_index", 0}, {"delta", text}});
            send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", id}, {"content_index", 0},
                        {"transcript", text}, {"usage", transcription_usage(asr_in, asr_out)}});
            items.clear();
            return true;
        }
        // transcription: the committed audio is transcribed on its own, then the slot is cleared for the next one
        {
            std::lock_guard<std::mutex> lock(mutex);
            transcribe_item_id = last_item_id;
            transcript.clear();
        }
        server_control_response response;
        response.n_predict = max_output_tokens;
        response.out_mode  = MTMD_HELPER_GEN_AUDIO_OUT_TEXT;
        response.sampling  = response_sampling(nullptr);
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, user_footer + assistant_header, {}, response)) {
            return false;
        }
        control(SERVER_CONTROL_ACTION_CLEAR);
        turn_open = false;
        n_turns   = 0;
        std::lock_guard<std::mutex> lock(mutex);
        items.clear();
        return true;
    }

    // a client that interrupts the assistant cancels the response and truncates it right after: the cancelled
    // response ends within a few frames (its item is in the conversation then), the edit waits for it
    void wait_cancelled_response() {
        for (int i = 0; i < 300 && responding && cancel_sent && !fault_pending; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // edits of the conversation wait for the response in progress and the input audio buffer; with_audio: the edit
    // goes around the open buffer (edit_around_audio)
    bool can_edit(const std::string & what, bool with_audio = false) {
        wait_cancelled_response();
        if (responding) {
            send_error(what + ": a response is in progress, cancel it or wait for response.done");
            return false;
        }
        if (audio_open && !with_audio) {
            send_error(what + ": the input audio buffer has audio, commit or clear it first");
            return false;
        }
        return true;
    }

    // an edit while the user speaks (e.g. the truncate of an interrupted answer): the audio of the open buffer leaves
    // the slot, the edit runs, then the buffer opens again after the edited conversation with the same audio
    bool edit_around_audio(const std::function<bool()> & edit) {
        if (!audio_open) {
            return edit();
        }
        if (!control(SERVER_CONTROL_ACTION_ROLLBACK, pending_mark)) {
            send_error("failed to edit the conversation", "server_error");
            return false;
        }
        load_turn(pending_before);
        audio_buf held = take_audio();
        const bool ok = edit();
        return reopen_audio(std::move(held)) && ok;
    }

    struct audio_buf {
        bool                             open = false;
        std::vector<float>               pcm;
        std::shared_ptr<server_asr_turn> asr; // cascade: the transcription of the buffer goes on as it is
    };

    // the open input audio buffer leaves the conversation state (the slot does not hold it any more)
    audio_buf take_audio() {
        audio_buf b{audio_open, std::move(pending_pcm), std::move(asr_turn)};
        audio_open = false;
        pending_pcm.clear();
        asr_turn.reset();
        return b;
    }

    // and opens again after the conversation as it is now, with the same audio
    bool reopen_audio(audio_buf b) {
        if (!b.open) {
            return true;
        }
        if (!begin_item() || !ensure_turn()) {
            return false;
        }
        audio_open  = true;
        pending_pcm = std::move(b.pcm);
        asr_turn    = std::move(b.asr);
        return asr_turn || pending_pcm.empty() || control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, pending_pcm);
    }

    // the input audio buffer is empty again, what the slot holds of it is gone already
    void reset_audio() {
        audio_open  = false;
        has_audio   = false;
        buf_samples = 0;
        pending_pcm.clear();
        if (asr_turn) {
            deps.asr->cancel(asr_turn);
            asr_turn.reset();
        }
        speech_item_id.clear();
        in_rs.reset(in_fmt.rate, model_in_rate);
        reset_detector();
    }

    void reset_detector() {
        detector.reset();
        if (turn_params) {
            detector = std::make_unique<server_turn_detector>(*turn_params, make_vad(), turn_model_for_session());
        }
    }

    // the helpers below need the mutex held
    static int find_in(const std::vector<conv_item> & list, const std::string & id) {
        for (size_t i = 0; i < list.size(); i++) {
            if (list[i].id == id) {
                return (int) i;
            }
        }
        return -1;
    }

    // an item in the context
    int find_item(const std::string & id) const {
        return find_in(items, id);
    }

    // an item of the conversation, in the context or not
    conv_item * any_item(const std::string & id) {
        const int i = find_in(items, id);
        const int o = i < 0 ? find_in(old_items, id) : -1;
        return i >= 0 ? &items[i] : o >= 0 ? &old_items[o] : nullptr;
    }

    std::string last_id() const {
        return !items.empty() ? items.back().id : !old_items.empty() ? old_items.back().id : "";
    }

    void send_added(const json & item, const json & prev) {
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", item}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", item}});
    }

    // takes the items from k on out of the slot, they stay in the conversation until put_stale()
    bool unput_from(size_t k) {
        std::string mark;
        turn_state  before;
        {
            std::lock_guard<std::mutex> lock(mutex);
            // a response goes back as a whole, from its first item
            while (k > 0 && k < items.size() && items[k].mark.empty()) {
                k--;
            }
            if (k >= stale_from || k >= items.size()) {
                return true;
            }
            mark   = items[k].mark;
            before = items[k].before;
        }
        drop_snapshots(k);
        if (!control(SERVER_CONTROL_ACTION_ROLLBACK, mark)) {
            send_error("failed to go back in the conversation", "server_error");
            return false;
        }
        load_turn(before);
        stale_from = k;
        return true;
    }

    // the items that left the slot (unput_from) go in again
    bool put_stale() {
        if (stale_from == SIZE_MAX) {
            return true;
        }
        std::vector<conv_item> tail;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const size_t k = std::min(stale_from, items.size());
            tail.assign(std::make_move_iterator(items.begin() + k), std::make_move_iterator(items.end()));
            items.erase(items.begin() + k, items.end());
            last_item_id = last_id();
        }
        stale_from = SIZE_MAX;
        // the transcriptions on other slots wait, as for a response
        const bool was_hot = hot;
        set_hot(true);
        const bool ok = put_items(tail, true);
        if (!was_hot) {
            set_hot(false);
        }
        return ok;
    }

    // goes back to item k and puts items k.. in again, after `change` edited them
    bool rebuild(size_t k, const std::function<void(std::vector<conv_item> &)> & change) {
        std::vector<conv_item> tail;
        std::string            mark;
        turn_state             before = save_turn();
        bool                   changed = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            // a response goes back as a whole, from its first item
            while (k > 0 && k < items.size() && items[k].mark.empty()) {
                k--;
            }
            if (k < items.size()) {
                changed = true;
                mark   = items[k].mark;
                before = items[k].before;
                tail.assign(std::make_move_iterator(items.begin() + k), std::make_move_iterator(items.end()));
                items.erase(items.begin() + k, items.end());
            }
            last_item_id = last_id();
        }
        if (changed) {
            drop_snapshots(k);
        }
        if (!mark.empty()) {
            if (!control(SERVER_CONTROL_ACTION_ROLLBACK, mark)) {
                send_error("failed to go back in the conversation", "server_error");
                return false;
            }
            load_turn(before);
        }
        change(tail);
        return put_items(tail, true);
    }

    // puts items in order, a response as one assistant turn; keep: they become the items of the conversation
    bool put_items(std::vector<conv_item> & tail, bool keep) {
        for (size_t i = 0; i < tail.size(); ) {
            if (tail[i].kind == conv_item::ASSISTANT) {
                std::vector<conv_item> group;
                const std::string resp = tail[i].response_id;
                while (i < tail.size() && tail[i].kind == conv_item::ASSISTANT && tail[i].response_id == resp) {
                    group.push_back(std::move(tail[i++]));
                }
                if (!put_response(group)) {
                    send_error("failed to rebuild the conversation", "server_error");
                    return false;
                }
                for (auto & it : group) {
                    if (keep) {
                        add_item(std::move(it), false);
                    }
                }
                continue;
            }
            if (!put_item(tail[i])) {
                send_error("failed to rebuild the conversation", "server_error");
                return false;
            }
            if (keep) {
                add_item(std::move(tail[i]), false);
            }
            i++;
        }
        return true;
    }

    // what a context is told before the conversation: its snapshots are reused for the same instructions and tools
    std::string context_key() const {
        return instructions + '\x1f' + tools_json.dump();
    }

    static std::vector<std::string> ids_of(const std::vector<conv_item> & list) {
        std::vector<std::string> ids;
        ids.reserve(list.size());
        for (const auto & it : list) {
            ids.push_back(it.id);
        }
        return ids;
    }

    static bool starts_with(const std::vector<std::string> & ids, const std::vector<std::string> & prefix) {
        return prefix.size() <= ids.size() && std::equal(prefix.begin(), prefix.end(), ids.begin());
    }

    // the snapshot of this context that holds the most items of the conversation
    ctx_snapshot * find_snapshot(const std::string & key, const std::vector<std::string> & ids) {
        ctx_snapshot * best = nullptr;
        for (auto & s : snapshots) {
            if (s.key == key && starts_with(ids, s.ids) && (!best || s.ids.size() > best->ids.size())) {
                best = &s;
            }
        }
        return best;
    }

    // keeps the own context in the slot (rolled back to its items) for the next response with the same context
    void keep_own_context() {
        if (own_ctx.key.empty() || !control(SERVER_CONTROL_ACTION_ROLLBACK, own_end_mark)) {
            own_ctx = {};
            return;
        }
        for (auto it = snapshots.begin(); it != snapshots.end(); ) {
            if (it->key == own_ctx.key) {
                control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, it->name);
                it = snapshots.erase(it);
            } else {
                ++it;
            }
        }
        if (snapshots.size() >= max_snapshots) {
            auto lru = std::min_element(snapshots.begin(), snapshots.end(),
                                        [](const ctx_snapshot & a, const ctx_snapshot & b) { return a.used < b.used; });
            control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, lru->name);
            snapshots.erase(lru);
        }
        own_ctx.name = "ctx" + std::to_string(n_snaps++);
        own_ctx.used = ++n_snap_use;
        if (control(SERVER_CONTROL_ACTION_SNAPSHOT_SAVE, own_ctx.name)) {
            snapshots.push_back(std::move(own_ctx));
        }
        own_ctx = {};
    }

    // items from k on change: the snapshots that hold them are of no use any more
    void drop_snapshots(size_t k) {
        for (auto it = snapshots.begin(); it != snapshots.end(); ) {
            if (it->ids.size() > k) {
                control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, it->name);
                it = snapshots.erase(it);
            } else {
                ++it;
            }
        }
    }

    // after a response with its own context: the session context from its snapshot, the items that came since go in
    // when the slot is needed (put_stale)
    bool restore_from_snapshot() {
        keep_own_context();
        const ctx_snapshot s = std::move(session_snap);
        session_snap = {};
        size_t n_items;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!starts_with(ids_of(items), s.ids)) {
                control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, s.name);
                return false;
            }
            n_items = items.size();
        }
        if (!control(SERVER_CONTROL_ACTION_SNAPSHOT_LOAD, s.name)) {
            return false;
        }
        control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, s.name);
        load_turn(s.turn);
        stale_from = s.ids.size() < n_items ? s.ids.size() : SIZE_MAX;
        return true;
    }

    // after a response with its own context: the session instructions and tools, the conversation as it is
    void restore_conversation() {
        if (snap_restore) {
            snap_restore    = false;
            pending_restore = false;
            instructions = saved_instructions;
            tools        = saved_tools;
            tools_json   = saved_tools_json;
            configure();
            if (restore_from_snapshot()) {
                return;
            }
            // the conversation changed under the snapshot: all of it goes in again
            drop_snapshots(0);
            put_conversation();
            return;
        }
        pending_restore = false;
        instructions = saved_instructions;
        tools        = saved_tools;
        tools_json   = saved_tools_json;
        configure();
        if (!control(SERVER_CONTROL_ACTION_ROLLBACK, restore_mark)) {
            send_error("failed to restore the conversation", "server_error");
            return;
        }
        load_turn(restore_before);
        std::vector<conv_item> all = take_items();
        put_items(all, true);
    }

    // the items of the context leave the conversation, to go into the slot again
    std::vector<conv_item> take_items() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<conv_item> all = std::move(items);
        items.clear();
        last_item_id = last_id();
        return all;
    }

    // the whole conversation goes into the slot again
    void put_conversation() {
        control(SERVER_CONTROL_ACTION_CLEAR);
        load_turn({});
        stale_from = SIZE_MAX;
        std::vector<conv_item> all = take_items();
        put_items(all, true);
    }

    void on_item_retrieve(const json & ev) {
        const std::string id = json_value(ev, "item_id", std::string());
        std::lock_guard<std::mutex> lock(mutex);
        const conv_item * it = any_item(id);
        if (!it) {
            send_error("conversation.item.retrieve: no item " + id, "invalid_request_error", "item_retrieve_invalid_item_id");
            return;
        }
        // audio content comes with its audio, in the format the client sent or got it
        json item = it->item;
        if (!it->audio.empty() && item.contains("content")) {
            for (auto & part : item["content"]) {
                const std::string t = json_value(part, "type", std::string());
                if (t == "input_audio" || t == "output_audio") {
                    part["audio"]  = base64::encode(it->audio);
                    part["format"] = it->audio_fmt.to_json();
                }
            }
        }
        send_event({{"type", "conversation.item.retrieved"}, {"item", item}});
    }

    void on_item_delete(const json & ev) {
        wait_cancelled_response();
        const std::string id = json_value(ev, "item_id", std::string());
        int i;
        {
            std::lock_guard<std::mutex> lock(mutex);
            i = find_item(id);
            const int o = i < 0 ? find_in(old_items, id) : -1;
            if (o >= 0) {
                // out of the context already: the slot does not change
                old_items.erase(old_items.begin() + o);
                last_item_id = last_id();
                send_event({{"type", "conversation.item.deleted"}, {"item_id", id}});
                return;
            }
        }
        if (i < 0) {
            send_error("conversation.item.delete: no item " + id, "invalid_request_error", "item_delete_invalid_item_id");
            return;
        }
        if (!can_edit("conversation.item.delete", true)) {
            return;
        }
        if (!audio_open) {
            if (!unput_from(i)) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            items.erase(std::remove_if(items.begin(), items.end(), [&](const conv_item & it) { return it.id == id; }), items.end());
            last_item_id = last_id();
            send_event({{"type", "conversation.item.deleted"}, {"item_id", id}});
            return;
        }
        const bool ok = edit_around_audio([&]() { return rebuild(i, [&](std::vector<conv_item> & tail) {
            tail.erase(std::remove_if(tail.begin(), tail.end(), [&](const conv_item & it) { return it.id == id; }), tail.end());
        }); });
        if (ok) {
            send_event({{"type", "conversation.item.deleted"}, {"item_id", id}});
        }
    }

    // the assistant audio was heard up to audio_end_ms: the transcript keeps what was said before that
    void on_item_truncate(const json & ev) {
        wait_cancelled_response();
        const std::string id    = json_value(ev, "item_id", std::string());
        const int         part  = json_value(ev, "content_index", 0);
        const int64_t     end_ms = json_value(ev, "audio_end_ms", (int64_t) -1);
        int i;
        bool found;
        int64_t n_samples = 0;
        int rate = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            i = find_item(id);
            const conv_item * it = any_item(id);
            found = it != nullptr;
            if (it) {
                n_samples = it->audio_samples;
                rate      = it->audio_rate;
            }
        }
        if (!found) {
            send_error("conversation.item.truncate: no item " + id, "invalid_request_error", "item_truncate_invalid_item_id");
            return;
        }
        if (rate == 0 || part != 0) {
            send_error("conversation.item.truncate: only the audio of an assistant message can be truncated", "invalid_request_error",
                       "unsupported_content_type");
            return;
        }
        if (end_ms < 0 || end_ms * rate / 1000 > n_samples) {
            send_error("conversation.item.truncate: audio_end_ms is beyond the audio of the item", "invalid_request_error", "invalid_value");
            return;
        }
        bool ok = false;
        if (i < 0) {
            // out of the context: the slot does not change
            std::lock_guard<std::mutex> lock(mutex);
            if (conv_item * it = any_item(id)) {
                truncate_audio(*it, end_ms);
                ok = true;
            }
        } else if (can_edit("conversation.item.truncate", true)) {
            ok = edit_around_audio([&]() { return rebuild(i, [&](std::vector<conv_item> & tail) {
                for (auto & it : tail) {
                    if (it.id == id) {
                        truncate_audio(it, end_ms);
                    }
                }
            }); });
        }
        if (ok) {
            send_event({{"type", "conversation.item.truncated"}, {"item_id", id}, {"content_index", part}, {"audio_end_ms", end_ms}});
        }
    }

    // the transcript of an assistant item is cut in proportion to the audio heard, at a word boundary
    static void truncate_audio(conv_item & it, int64_t end_ms) {
        json & content = it.item["content"];
        std::string tr = content.size() > 0 ? json_value(content[0], "transcript", std::string()) : std::string();
        const int64_t end_s = end_ms * it.audio_rate / 1000;
        size_t keep = it.audio_samples > 0 ? (size_t) ((double) tr.size() * end_s / it.audio_samples) : tr.size();
        if (!it.spoken.empty()) {
            // a voice spoke the text piece by piece: whole pieces up to the end, the piece in progress in proportion
            int64_t s0 = 0, t0 = 0;
            keep = tr.size();
            for (const auto & [s1, t1] : it.spoken) {
                if (end_s < s1) {
                    keep = (size_t) (t0 + (t1 - t0) * (end_s - s0) / std::max<int64_t>(1, s1 - s0));
                    break;
                }
                s0 = s1;
                t0 = t1;
            }
        }
        if (keep < tr.size()) {
            const size_t sp = tr.rfind(' ', keep);
            keep = sp == std::string::npos ? 0 : sp;
        }
        if (content.size() > 0) {
            content[0]["transcript"] = ""; // as OpenAI: no text the user did not hear
        }
        it.text          = tr.substr(0, keep); // what goes back into the conversation
        it.audio_samples = end_s;
        const size_t bps = it.audio_fmt.type == "audio/pcm" ? 2 : 1;
        it.audio.resize(std::min(it.audio.size(), (size_t) it.audio_samples * bps));
    }

    // the audio not committed yet leaves the conversation
    void on_clear() {
        if (audio_open) {
            if (!control(SERVER_CONTROL_ACTION_ROLLBACK, pending_mark)) {
                send_error("failed to clear the input audio buffer", "server_error");
                return;
            }
            load_turn(pending_before);
        }
        reset_audio();
        send_event({{"type", "input_audio_buffer.cleared"}});
    }

    // an item from the client, as it goes into the conversation; path: of the item in the client event
    // false: err says why, or it is empty and the error is sent already (unknown field)
    bool parse_client_item(const json & item, const std::string & path, conv_item & it, std::string & err) {
        const std::string type = json_value(item, "type", std::string("message"));
        const std::string role = json_value(item, "role", std::string());
        if (!known_fields(item, path, type == "function_call_output" ? std::vector<const char *>{"id", "type", "status", "call_id", "output"} :
                                      type == "function_call" ? std::vector<const char *>{"id", "type", "status", "name", "call_id", "arguments"} :
                                                                std::vector<const char *>{"id", "type", "status", "role", "content"})) {
            return false;
        }
        it.id = item.contains("id") && item["id"].is_string() ? item["id"].get<std::string>() : "item_" + random_string();
        if (type == "function_call_output") {
            it.kind = conv_item::TOOL_OUTPUT;
            it.text = json_value(item, "output", std::string());
            it.item = {{"id", it.id}, {"type", type}, {"call_id", json_value(item, "call_id", std::string())}, {"output", it.text}};
            return true;
        }
        if (type == "function_call") {
            it.kind        = conv_item::ASSISTANT;
            it.response_id = "client_" + it.id;
            it.item = {{"id", it.id}, {"type", type}, {"status", "completed"}, {"name", json_value(item, "name", std::string())},
                       {"call_id", json_value(item, "call_id", std::string())}, {"arguments", json_value(item, "arguments", std::string())}};
            return true;
        }
        if (type != "message" || (role != "user" && role != "assistant" && role != "system")) {
            err = "only user, assistant and system messages, function calls and function call outputs are supported";
            return false;
        }
        json content = json::array();
        std::string text;
        const json parts_in = item.value("content", json::array());
        for (size_t i = 0; parts_in.is_array() && i < parts_in.size(); i++) {
            const json & p = parts_in[i];
            const std::string t = json_value(p, "type", std::string());
            if (!known_fields(p, path + ".content[" + std::to_string(i) + "]",
                              t == "input_audio" ? std::vector<const char *>{"type", "audio", "transcript"} : std::vector<const char *>{"type", "text"})) {
                return false;
            }
            if ((role == "user" || role == "system") && t == "input_text") {
                text += json_value(p, "text", std::string());
                content.push_back({{"type", t}, {"text", json_value(p, "text", std::string())}});
            } else if (role == "user" && t == "input_audio") {
                std::string bytes;
                try {
                    bytes = base64::decode(json_value(p, "audio", std::string()));
                } catch (const std::exception &) {
                    err = "invalid base64 audio";
                    return false;
                }
                const std::vector<float> x = decode_audio(bytes, in_fmt);
                it.audio     += bytes;
                it.audio_fmt  = in_fmt;
                pcm_resampler rs;
                rs.reset(in_fmt.rate, model_in_rate);
                it.pcm = rs.process(x.data(), x.size());
                const std::vector<float> tail = rs.flush();
                it.pcm.insert(it.pcm.end(), tail.begin(), tail.end());
                content.push_back({{"type", t}, {"transcript", p.value("transcript", json(nullptr))}});
            } else if (role == "assistant" && (t == "output_text" || t == "text")) {
                text += json_value(p, "text", std::string());
                content.push_back({{"type", "output_text"}, {"text", json_value(p, "text", std::string())}});
            } else {
                err = "unsupported content type " + t + " for role " + role;
                return false;
            }
        }
        if (content.empty()) {
            err = "the content of a message must not be empty";
            return false;
        }
        if (!it.pcm.empty() && !text.empty()) {
            err = "a user message has either text or audio";
            return false;
        }
        it.kind = role == "assistant" ? conv_item::ASSISTANT : role == "system" ? conv_item::SYSTEM :
                  it.pcm.empty() ? conv_item::USER_TEXT : conv_item::USER_AUDIO;
        it.text = text;
        if (it.kind == conv_item::ASSISTANT) {
            it.response_id = "client_" + it.id;
        }
        it.item = {{"id", it.id}, {"type", "message"}, {"status", "completed"}, {"role", role}, {"content", content}};
        return true;
    }

    void on_item_create(const json & ev) {
        if (type == "transcription") {
            send_error("conversation.item.create: not supported in a transcription session");
            return;
        }
        conv_item it;
        std::string err;
        if (!parse_client_item(ev.value("item", json::object()), "item", it, err)) {
            if (!err.empty()) {
                send_error("conversation.item.create: " + err, "invalid_request_error", "invalid_value");
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (any_item(it.id)) {
                send_error("conversation.item.create: an item with id " + it.id + " already exists", "invalid_request_error",
                           "item_create_duplicate_item_id");
                return;
            }
            if (it.kind == conv_item::TOOL_OUTPUT) {
                const std::string call_id = json_value(it.item, "call_id", std::string());
                if (find_call(items, call_id) < 0 && find_call(old_items, call_id) < 0) {
                    send_error("conversation.item.create: no tool call " + call_id + " in the conversation", "invalid_request_error",
                               "invalid_tool_call_id");
                    return;
                }
            }
        }
        // a tool output at the end joins the items that are not in the slot, see on_tool_output()
        if (it.kind != conv_item::TOOL_OUTPUT && !put_stale()) {
            return;
        }
        // previous_item_id: insert after that item ("root": before the first one), the items after it go in again
        if (ev.contains("previous_item_id") && ev["previous_item_id"].is_string() && ev["previous_item_id"] != last_item_id) {
            const std::string prev = ev["previous_item_id"];
            if (prev != "root") {
                std::lock_guard<std::mutex> lock(mutex);
                if (!any_item(prev)) {
                    send_error("conversation.item.create: no item " + prev, "invalid_request_error", "item_create_invalid_previous_item_id");
                    return;
                }
            }
            insert_item(std::move(it), prev);
            return;
        }
        // audio not committed yet stays in the buffer, the item goes in before it
        edit_around_audio([&]() { return append_client_item(std::move(it)); });
    }

    // the function call with this call_id in a list of items (mutex held)
    static int find_call(const std::vector<conv_item> & list, const std::string & call_id) {
        for (size_t i = 0; i < list.size(); i++) {
            if (json_value(list[i].item, "type", std::string()) == "function_call" && json_value(list[i].item, "call_id", std::string()) == call_id) {
                return (int) i;
            }
        }
        return -1;
    }

    // a new item after the item `prev` ("root": before the first one), the items after it go in again
    bool insert_item(conv_item it, const std::string & prev) {
        const json out      = it.item;
        const json prev_out = prev == "root" ? json(nullptr) : json(prev);
        int i = -1;
        {
            std::lock_guard<std::mutex> lock(mutex);
            i = prev == "root" ? -1 : find_item(prev);
            const int o = prev == "root" || i >= 0 ? -1 : find_in(old_items, prev);
            if (prev == "root" ? !old_items.empty() : o >= 0 && o + 1 < (int) old_items.size()) {
                // among the items out of the context: it stays out with them, the slot does not change
                old_items.insert(old_items.begin() + o + 1, std::move(it));
                send_added(out, prev_out);
                return true;
            }
        }
        if (!can_edit("conversation.item.create", true)) {
            return false;
        }
        const bool ok = edit_around_audio([&]() {
            return rebuild(i + 1, [&](std::vector<conv_item> & tail) { tail.insert(tail.begin(), std::move(it)); });
        });
        if (ok) {
            send_added(out, prev_out);
        }
        return ok;
    }

    // a new item at the end of the conversation
    bool append_client_item(conv_item it) {
        std::string err;
        if (it.kind == conv_item::SYSTEM) {
            system_text_here(err);
            if (!err.empty()) {
                send_error("conversation.item.create: " + err);
                return false;
            }
        }
        if (it.kind == conv_item::TOOL_OUTPUT) {
            return on_tool_output(std::move(it));
        }
        if (it.kind == conv_item::ASSISTANT) {
            std::vector<conv_item> group;
            group.push_back(std::move(it));
            if (!put_response(group)) {
                return false;
            }
            add_item(std::move(group[0]));
            return true;
        }
        if (!put_item(it)) {
            return false;
        }
        const std::string id = it.id;
        std::vector<float> pcm = it.kind == conv_item::USER_AUDIO ? it.pcm : std::vector<float>();
        const std::string text = it.transcript;
        add_item(std::move(it));
        if (pcm.empty()) {
            return true;
        }
        if (!cascade()) {
            transcribe_item(id, std::move(pcm));
        } else if (type == "realtime" && !transcription.is_null()) {
            std::lock_guard<std::mutex> lock(mutex);
            send_transcription(id, true, text, 0, 0); // put_item transcribed it
        }
        return true;
    }

    // the result of a tool call: the chat template has it right after the response that called the tool, so it goes
    // there, after the results that response has already, also when other items came since
    bool on_tool_output(conv_item it) {
        if (!has_tool_text) {
            send_error("conversation.item.create: the chat template has no tool results");
            return false;
        }
        const std::string call_id = json_value(it.item, "call_id", std::string());
        // the last item of the response with the call and of its results
        auto end_of = [](const std::vector<conv_item> & list, size_t c) {
            size_t e = c;
            while (e + 1 < list.size() && list[e + 1].kind == conv_item::ASSISTANT && list[e + 1].response_id == list[c].response_id) {
                e++;
            }
            while (e + 1 < list.size() && list[e + 1].kind == conv_item::TOOL_OUTPUT) {
                e++;
            }
            return e;
        };
        std::string prev;
        bool        last = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const int c = find_call(items, call_id);
            const int o = c < 0 ? find_call(old_items, call_id) : -1;
            if (c >= 0) {
                const size_t e = end_of(items, c);
                prev = items[e].id;
                last = e + 1 == items.size();
            } else if (o >= 0) {
                // the call is out of the context: so is its result
                const size_t e = end_of(old_items, o);
                const json out = it.item;
                prev = old_items[e].id;
                old_items.insert(old_items.begin() + e + 1, std::move(it));
                send_added(out, prev);
                return true;
            } else {
                return false;
            }
        }
        if (!last) {
            return put_stale() && insert_item(std::move(it), prev);
        }
        if (stale_from != SIZE_MAX) {
            add_item(std::move(it)); // the slot gets it with the other items it lacks
            return true;
        }
        if (turn_open) {
            send_error("conversation.item.create: a function call output cannot go into an open user turn");
            return false;
        }
        if (!put_item(it)) {
            return false;
        }
        add_item(std::move(it));
        return true;
    }

    // the context of a response with its own input, instructions or tools: the conversation (or the input items) from
    // the start, rendered with them; restore_conversation() puts the conversation back once the response is done
    bool put_response_context(const json & resp, bool has_input, const std::string & instr,
                              const std::vector<common_chat_tool> & new_tools, const json & new_tools_json) {
        if (!can_edit("response.create")) {
            return false;
        }
        std::vector<conv_item> ctx_items;
        if (!has_input) {
            std::lock_guard<std::mutex> lock(mutex);
            ctx_items = items;
        }
        if (has_input) {
            for (size_t n = 0; n < resp["input"].size(); n++) {
                const json &      in   = resp["input"][n];
                const std::string path = "response.input[" + std::to_string(n) + "]";
                if (json_value(in, "type", std::string()) == "item_reference") {
                    if (!known_fields(in, path, {"type", "id"})) {
                        return false;
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    const conv_item * ref = any_item(json_value(in, "id", std::string()));
                    if (!ref) {
                        send_error("response.create: no item " + json_value(in, "id", std::string()), "invalid_request_error",
                                   "invalid_reference_item_id");
                        return false;
                    }
                    ctx_items.push_back(*ref);
                    ctx_items.back().response_id += "_ref"; // a referenced response item goes in on its own
                    continue;
                }
                conv_item it;
                std::string err;
                if (!parse_client_item(in, path, it, err)) {
                    if (!err.empty()) {
                        send_error("response.create: input: " + err, "invalid_request_error", "invalid_value");
                    }
                    return false;
                }
                ctx_items.push_back(std::move(it));
            }
        }

        saved_instructions = instructions;
        saved_tools        = tools;
        saved_tools_json   = tools_json;
        instructions = instr;
        tools        = new_tools;
        tools_json   = new_tools_json;
        if (!configure()) {
            instructions = saved_instructions;
            tools        = saved_tools;
            tools_json   = saved_tools_json;
            send_error("the chat template does not fit the realtime turn format", "server_error");
            return false;
        }
        if (!has_input) {
            return put_cached_context(ctx_items);
        }
        // the own context starts with its own instructions and tools: nothing of the conversation stays in the slot
        // (the mark of the first item is after the session prompt if the conversation starts with a response)
        if (!control(SERVER_CONTROL_ACTION_ROLLBACK)) {
            send_error("failed to go back in the conversation", "server_error");
            return false;
        }
        restore_mark   = "base" + std::to_string(n_marks++);
        restore_before = {};
        if (!control(SERVER_CONTROL_ACTION_MARK, restore_mark)) {
            return false;
        }
        pending_restore = true;
        load_turn({});
        if (!put_items(ctx_items, false)) {
            return false;
        }
        // a response needs a user turn or tool results before it
        return turn_open || tool_results || system_last || ensure_turn();
    }

    // the own context of a response over the conversation (no input of its own): the session context goes into a
    // snapshot, the own context starts from the snapshot of the same context that holds the most items, if any
    bool put_cached_context(std::vector<conv_item> & ctx_items) {
        // the session context may lack the last items (stale_from): they go in when it is back
        const std::vector<std::string> ids = ids_of(ctx_items);
        own_ctx           = {};
        session_snap      = {};
        session_snap.name = "session";
        session_snap.ids.assign(ids.begin(), ids.begin() + std::min(stale_from, ids.size()));
        session_snap.turn = save_turn();
        if (!control(SERVER_CONTROL_ACTION_SNAPSHOT_SAVE, session_snap.name)) {
            return false;
        }
        pending_restore = true;
        snap_restore    = true;

        const std::string key = context_key();
        size_t from = 0;
        if (ctx_snapshot * hit = find_snapshot(key, ids)) {
            if (!control(SERVER_CONTROL_ACTION_SNAPSHOT_LOAD, hit->name)) {
                return false;
            }
            hit->used = ++n_snap_use;
            load_turn(hit->turn);
            from = hit->ids.size();
        } else {
            if (!control(SERVER_CONTROL_ACTION_ROLLBACK)) {
                send_error("failed to go back in the conversation", "server_error");
                return false;
            }
            load_turn({});
        }
        std::vector<conv_item> rest(std::make_move_iterator(ctx_items.begin() + from), std::make_move_iterator(ctx_items.end()));
        if (!put_items(rest, false)) {
            return false;
        }
        own_ctx.key  = key;
        own_ctx.ids  = ids;
        own_ctx.turn = save_turn();
        own_end_mark = "own" + std::to_string(n_marks++);
        if (!control(SERVER_CONTROL_ACTION_MARK, own_end_mark)) {
            return false;
        }
        // a response needs a user turn or tool results before it
        return turn_open || tool_results || system_last || ensure_turn();
    }

    // from the start of a response to its end (with its speech) the transcriptions on other slots wait, they share the GPU
    void set_hot(bool on) {
        if (hot.exchange(on) == on) {
            return;
        }
        deps.n_hot += on ? 1 : -1;
        if (!on) {
            // the slots that waited have work again
            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = deps.queue_tasks.get_new_id();
            deps.queue_tasks.post(std::move(task));
        }
    }

    // from_client: the client asked for it, else the turn detection
    // again: the response in progress starts again after a slot error (recover), the client has its response.created
    // false: no response runs
    bool on_response_create(const json & ev, bool from_client = true, bool again = false) {
        set_hot(true);
        const bool ok = create_response(ev, from_client, again);
        if (!ok && !responding) {
            set_hot(false);
        }
        return ok;
    }

    bool create_response(const json & ev, bool from_client, bool again) {
        if (type == "transcription") {
            send_error("response.create: not supported in a transcription session");
            return false;
        }
        const json resp = ev.value("response", json::object());
        if (!again) {
            // one response at a time; a response the client just cancelled ends first
            wait_cancelled_response();
            if (responding) {
                if (from_client) {
                    send_error("response.create: the conversation already has an active response in progress: " + response_id,
                               "invalid_request_error", "conversation_already_has_active_response");
                }
                return false;
            }
            if (!response_fields(resp)) {
                return false;
            }
            resp_retries = 0;
            resp_tight   = false;
        }
        const std::string conversation = json_value(resp, "conversation", std::string("auto"));
        if (conversation != "auto" && conversation != "none") {
            send_error("response.create: conversation must be \"auto\" or \"none\"", "invalid_request_error", "invalid_value");
            return false;
        }
        const bool oob = conversation == "none";

        // input, instructions or tools of the response: it gets its own context
        std::string resp_instructions = instructions;
        if (resp.contains("instructions") && resp["instructions"].is_string()) {
            resp_instructions = resp["instructions"];
        }
        std::vector<common_chat_tool> resp_tools = tools;
        json resp_tools_json = tools_json;
        if (resp.contains("tools") && resp["tools"] != tools_json) {
            std::string err;
            if (!parse_tools(resp["tools"], resp_tools, err)) {
                send_error("response.create: " + err, "invalid_request_error", "invalid_value");
                return false;
            }
            resp_tools_json = resp["tools"];
        }
        const bool has_input = resp.contains("input") && resp["input"].is_array();
        const bool fresh     = has_input || resp_instructions != instructions || resp_tools_json != tools_json;

        mtmd_helper_gen_audio_out_mode mode = out_mode;
        if (resp.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(resp["output_modalities"], mode, err)) {
                send_error("response.create: " + err, "invalid_request_error", "invalid_value");
                return false;
            }
        }
        // audio.output of the response: its own format and voice
        audio_format new_resp_fmt;
        {
            std::lock_guard<std::mutex> lock(mutex);
            new_resp_fmt = out_fmt;
        }
        std::string resp_voice = voice;
        const json  resp_audio = resp.contains("audio") && resp["audio"].is_object() ? resp["audio"].value("output", json::object()) : json::object();
        if (resp_audio.is_object()) {
            std::string err;
            if (resp_audio.contains("format") && !parse_audio_format(resp_audio["format"], new_resp_fmt, err)) {
                send_error("response.create: audio.output." + err, "invalid_request_error", "invalid_value");
                return false;
            }
            if (resp_audio.contains("voice")) {
                const json & v = resp_audio["voice"];
                resp_voice = v.is_string() ? v.get<std::string>() : v.is_object() ? json_value(v, "id", voice) : voice;
            }
        }
        // with a voice model the session speaks by default, as the OpenAI API
        if (deps.voices && mode == MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            mode = MTMD_HELPER_GEN_AUDIO_OUT_AUDIO;
        }
        if (!can_speak() && mode != MTMD_HELPER_GEN_AUDIO_OUT_TEXT && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate audio, start the server with --voice-model");
            return false;
        }
        if (!deps.cap.text_gen && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate text");
            return false;
        }
        // audio not committed yet belongs to this turn
        if (!again && has_audio && !commit()) {
            return false;
        }
        if (ctx_lost) {
            // the conversation did not fit before: one more try, the client may have deleted items
            ctx_lost = false;
            audio_buf held = take_audio();
            reset_context();
            put_conversation();
            reopen_audio(std::move(held));
        }
        // a response with its own context over the conversation takes the items as they are, see put_cached_context()
        if ((!fresh || has_input) && !put_stale()) {
            return false;
        }
        if (fresh && !put_response_context(resp, has_input, resp_instructions, resp_tools, resp_tools_json)) {
            return false;
        }
        // an empty conversation: the assistant speaks first, after an empty user turn
        if (at_start() && !ensure_turn()) {
            return false;
        }

        common_chat_tool_choice choice = tool_choice;
        if (resp.contains("tool_choice")) {
            std::string err;
            if (!parse_tool_choice(resp["tool_choice"], choice, err)) {
                send_error("response.create: " + err, "invalid_request_error", "invalid_value");
                return false;
            }
        }

        // the gen-audio pipeline opens the response in the asked mode, after tool results the assistant turn opens right away
        // after a response without new input, the assistant turn follows the previous one
        const std::string prompt = (turn_open ? user_footer : tool_results ? "" : next_header) + assistant_header;
        server_control_response response;
        response.n_predict = max_output_tokens;
        if (resp.contains("max_output_tokens") && !parse_max_output_tokens(resp["max_output_tokens"], response.n_predict)) {
            send_error("response.create: max_output_tokens must be an integer or \"inf\"", "invalid_request_error", "invalid_value");
            return false;
        }
        response.n_limit = resp_tight ? 0 : response_limit(response.n_predict);
        // an audio response carries its transcript: the model speaks interleaved with the text it says, or a voice
        // model speaks the text of the model
        const bool use_voice       = deps.voices && (mode == MTMD_HELPER_GEN_AUDIO_OUT_AUDIO || mode == MTMD_HELPER_GEN_AUDIO_OUT_MIXED);
        const bool with_transcript = mode == MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && deps.cap.text_gen;
        response.out_mode = use_voice ? MTMD_HELPER_GEN_AUDIO_OUT_TEXT : with_transcript ? MTMD_HELPER_GEN_AUDIO_OUT_MIXED : mode;
        int resp_voice_model = 0;
        std::string resp_voice_name;
        if (use_voice) {
            resp_voice_model = std::max(0, deps.voices->find(resp_voice == "default" ? "" : resp_voice, resp_voice_name));
        }
        if (!use_voice) {
            response.lang  = out_language;
            response.voice = model_voice(resp_voice);
        }

        // the output goes through the parser of the chat template: content, reasoning and tool calls (with their grammar)
        std::optional<common_chat_session> chat;
        {
            common_chat_msg user;
            user.role    = "user";
            user.content = "@@USER@@";
            common_chat_templates_inputs inputs = chat_inputs({user}, true);
            inputs.tool_choice         = choice;
            inputs.parallel_tool_calls = true;
            try {
                chat.emplace(deps.meta.chat_params.tmpls.get(), deps.vocab, inputs);
            } catch (const std::exception & e) {
                if (!tools.empty()) {
                    send_error(std::string("response.create: ") + e.what(), "server_error");
                    return false;
                }
                chat.reset(); // without tools the raw text goes out
            }
        }
        response.sampling = response_sampling(chat ? &*chat : nullptr);

        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!again) {
                response_id      = "resp_" + random_string();
                response_item_id = "item_" + random_string();
            }
            response_text.clear();
            response_transcript.clear();
            transcript_mode = with_transcript;
            out_samples     = 0;
            out_audio.clear();
            voice_resp    = use_voice;
            voice_model   = resp_voice_model;
            voice_name    = resp_voice_name;
            segmenter     = server_voice_segmenter();
            voice_text    = 0;
            voice_pending = 0;
            voice_frames  = 0;
            llm_done      = false;
            held_done.reset();
            voice_cancel  = std::make_shared<std::atomic<bool>>(false);
            voice_segments.clear();
            voice_sent    = 0;
            parts       = json::array();
            idx_text    = -1;
            idx_audio   = -1;
            output      = json::array();
            idx_message = -1;
            call_items.clear();
            parse.reset();
            if (chat) {
                parse.emplace(std::move(*chat));
            }
            resp_fmt      = new_resp_fmt;
            resp_item_pos = items.size();
            out_rs.reset(use_voice ? deps.voices->models()[resp_voice_model].sample_rate : model_out_rate, resp_fmt.rate);
            resp_oob      = oob;
            response_meta = {
                {"conversation_id",   oob ? json(nullptr) : json(conv_id)},
                {"output_modalities", output_modalities_json(mode)},
                {"max_output_tokens", max_output_tokens_json(response.n_predict)},
                {"metadata",          resp.value("metadata", json(nullptr))},
            };
            if (can_speak()) {
                response_meta["audio"] = {{"output", {{"format", resp_fmt.to_json()}, {"voice", resp_voice}}}};
            }
            if (!again) {
                send_event({{"type", "response.created"}, {"response", response_json("in_progress", nullptr, nullptr)}});
            }
        }
        if (!begin_item()) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            resp_mark   = pending_mark;
            resp_before = pending_before;
        }
        if (!again) {
            resp_event       = ev;
            resp_from_client = from_client;
        }
        responding     = true;
        cancel_sent    = false;
        cancel_by_turn = false;
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, prompt, {}, response)) {
            responding = false;
            return false;
        }
        if (oob && !fresh) {
            // out of the conversation: back to the mark once the response is done, the slot runs the controls in order
            control(SERVER_CONTROL_ACTION_ROLLBACK, resp_mark);
            load_turn(resp_before);
            return true;
        }
        // the response ends the turn, its tokens are in the slot; the footer goes in before the next turn
        next_header  = assistant_footer;
        turn_open    = false;
        tool_results = false;
        system_last  = false;
        n_turns++;
        return true;
    }

    // the whole update is checked first: a rejected update changes nothing and gets only an error
    void on_session_update(const json & ev) {
        const json session = ev.value("session", json::object());
        auto reject = [this](const std::string & msg) { send_error("session.update: " + msg, "invalid_request_error", "invalid_value"); };
        if (!session.is_object()) {
            reject("session must be an object");
            return;
        }
        std::string err;

        std::string new_type = type;
        if (session.contains("type") && session["type"].is_string() && session["type"] != type) {
            new_type = session["type"];
            if (new_type != "realtime" && new_type != "transcription") {
                reject("unknown session type " + new_type);
                return;
            }
            if (started) {
                reject("the session type cannot change after the first audio or message");
                return;
            }
        }
        if (!session_fields(session, new_type == "realtime")) {
            return;
        }
        truncation_params new_trunc = trunc;
        if (session.contains("truncation") && !parse_truncation(session["truncation"], new_trunc, err)) {
            reject(err);
            return;
        }
        std::string new_instructions = instructions;
        if (session.contains("instructions") && session["instructions"].is_string()) {
            new_instructions = session["instructions"];
        }
        mtmd_helper_gen_audio_out_mode new_out_mode = out_mode;
        if (session.contains("output_modalities") && !parse_output_modalities(session["output_modalities"], new_out_mode, err)) {
            reject(err);
            return;
        }
        std::vector<common_chat_tool> new_tools = tools;
        json new_tools_json = tools_json;
        if (session.contains("tools") && session["tools"] != tools_json) {
            if (!parse_tools(session["tools"], new_tools, err)) {
                reject(err);
                return;
            }
            new_tools_json = session["tools"];
        }
        common_chat_tool_choice new_choice = tool_choice;
        if (session.contains("tool_choice") && !parse_tool_choice(session["tool_choice"], new_choice, err)) {
            reject(err);
            return;
        }
        const json new_choice_json = session.contains("tool_choice") ? session["tool_choice"] : tool_choice_json;
        int32_t new_max = max_output_tokens;
        if (session.contains("max_output_tokens") && !parse_max_output_tokens(session["max_output_tokens"], new_max)) {
            reject("max_output_tokens must be an integer or \"inf\"");
            return;
        }

        const json audio = session.value("audio", json::object());
        const json in    = audio.is_object() ? audio.value("input", json::object()) : json::object();
        const json out   = audio.is_object() ? audio.value("output", json::object()) : json::object();
        audio_format new_in_fmt  = in_fmt;
        audio_format new_out_fmt = out_fmt;
        if (in.contains("format") && !parse_audio_format(in["format"], new_in_fmt, err)) {
            reject("audio.input." + err);
            return;
        }
        if (out.contains("format") && !parse_audio_format(out["format"], new_out_fmt, err)) {
            reject("audio.output." + err);
            return;
        }
        std::optional<server_turn_params> new_turn = turn_params;
        if (in.contains("turn_detection") && !parse_turn_detection(in["turn_detection"], new_turn, err)) {
            reject("audio.input." + err);
            return;
        }
        json new_transcription = transcription;
        std::string new_language = language;
        if (in.contains("transcription")) {
            const json & t = in["transcription"];
            if (!t.is_null() && !t.is_object()) {
                reject("audio.input.transcription must be an object or null");
                return;
            }
            new_transcription = t.is_null() ? json(nullptr) : json{{"model", json_value(t, "model", std::string())},
                                                                   {"language", t.value("language", json(nullptr))},
                                                                   {"prompt", t.value("prompt", json(nullptr))}};
            new_language = t.is_object() ? json_value(t, "language", std::string()) : std::string();
        }
        if (new_type == "transcription" && new_language != language && started) {
            reject("the transcription language cannot change after the first audio");
            return;
        }

        // a string or {"id": name}: a voice model (--voice-model), else a voice of a model that speaks itself
        std::string new_voice = voice;
        if (out.contains("voice")) {
            const json & v = out["voice"];
            new_voice = v.is_string() ? v.get<std::string>() : v.is_object() ? json_value(v, "id", voice) : voice;
        }
        const std::string new_out_language = json_value(out, "language", out_language);
        const bool speech_changed = speaks_itself() && speech_prompt(new_out_language, new_voice) != speech_prompt(out_language, voice);

        // new instructions or tools change the text around every turn: the conversation is rebuilt with them
        // so does a voice or language that goes into the system prompt
        const bool rerender = started && (new_instructions != instructions || new_tools_json != tools_json || speech_changed);
        if (rerender && !can_edit("session.update")) {
            return;
        }

        // checked: apply
        const auto old_type = type, old_instructions = instructions, old_language = language;
        const auto old_voice = voice, old_out_language = out_language;
        const auto old_tools = tools;
        type         = new_type;
        instructions = new_instructions;
        language     = new_language;
        voice        = new_voice;
        out_language = new_out_language;
        tools        = new_tools;
        if ((!started || rerender) && !configure()) {
            type         = old_type;
            instructions = old_instructions;
            language     = old_language;
            voice        = old_voice;
            out_language = old_out_language;
            tools        = old_tools;
            send_error("the chat template does not fit the realtime turn format", "server_error");
            return;
        }
        tools_json        = new_tools_json;
        out_mode          = new_out_mode;
        tool_choice       = new_choice;
        tool_choice_json  = new_choice_json;
        max_output_tokens = new_max;
        transcription     = new_transcription;
        if (in.contains("noise_reduction")) {
            noise_reduction = in["noise_reduction"];
        }
        if (out.contains("speed") && out["speed"].is_number()) {
            speed = out["speed"].get<double>();
        }
        truncation = session.value("truncation", truncation);
        trunc      = new_trunc;
        tracing    = session.value("tracing", tracing);
        include    = session.value("include", include);
        prompt     = session.value("prompt", prompt);
        if (new_in_fmt.type != in_fmt.type || new_in_fmt.rate != in_fmt.rate) {
            in_fmt = new_in_fmt;
            in_rs.reset(in_fmt.rate, model_in_rate);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            out_fmt = new_out_fmt;
        }
        if (in.contains("turn_detection")) {
            turn_params = new_turn;
            reset_detector();
        }
        if (rerender && !rebuild(0, [](std::vector<conv_item> &) {})) {
            return;
        }
        send_event({{"type", "session.updated"}, {"session", session_json()}});
    }

    // turn text of the session type, before anything went into the slot
    bool configure() {
        const std::string saved = instructions;
        if (type == "transcription") {
            const auto preset = common_chat_get_asr_prompt(deps.meta.chat_params.tmpls.get(), language);
            instructions = preset.system;
        } else if (speaks_itself()) {
            const std::string speech = speech_prompt(out_language, voice);
            if (!speech.empty()) {
                instructions = instructions.empty() ? speech : speech + "\n" + instructions;
            }
        }
        const std::string key = context_key();
        bool ok;
        if (auto it = turn_texts.find(key); it != turn_texts.end()) {
            ok = load_turn_text(it->second);
        } else {
            ok = derive_turn_text();
            if (instructions != saved) {
                assistant_open.clear(); // render_assistant() renders with the instructions of the session
            }
            if (turn_texts.size() >= max_turn_texts) {
                turn_texts.clear();
            }
            turn_texts[key] = save_turn_text(ok);
        }
        instructions = saved;
        return ok;
    }

    void handle(const std::string & msg) {
        json ev;
        try {
            ev = json::parse(msg);
        } catch (const std::exception &) {
            send_error("invalid JSON", "invalid_request_error", "invalid_json");
            return;
        }
        const std::string t = ev.is_object() ? json_value(ev, "type", std::string()) : std::string();
        if (pending_restore) {
            if (responding && t == "input_audio_buffer.append" && detector) {
                on_append(ev); // the turn detection runs now, so that speech interrupts the response (on_turn_audio)
                return;
            }
            if (responding && t != "response.cancel") {
                held_events.push_back({msg, {}}); // runs on the conversation once it is back
                return;
            }
            if (!responding) {
                set_hot(true);
                restore_conversation();
                run_held();
                if (!responding) {
                    set_hot(false);
                }
            }
        }
        // the items that left the slot go in again when it is needed, or with the audio after other events (idle time);
        // deletes, tool outputs and a response with its own context leave the slot as it is
        const bool append = t == "input_audio_buffer.append";
        if (append ? last_append : t != "conversation.item.delete" && t != "response.create" && t != "conversation.item.create") {
            put_stale();
        }
        last_append = append;
        client_event_id = ev.is_object() ? json_value(ev, "event_id", std::string()) : std::string();
        dispatch(t, ev);
        client_event_id.clear();
    }

    // the client events that waited for a response with its own context
    void run_held() {
        std::deque<held_event> held;
        held.swap(held_events);
        for (auto & h : held) {
            if (!h.msg.empty()) {
                handle(h.msg);
            } else if (holding()) {
                held_events.push_back(std::move(h)); // a replayed event started another such response
            } else {
                run_turn_actions(h.actions);
            }
        }
    }

    void dispatch(const std::string & t, const json & ev) {
        static const std::map<std::string, std::vector<const char *>> fields = {
            {"session.update",               {"type", "event_id", "session"}},
            {"transcription_session.update", {"type", "event_id", "session"}},
            {"input_audio_buffer.append",    {"type", "event_id", "audio"}},
            {"input_audio_buffer.commit",    {"type", "event_id"}},
            {"input_audio_buffer.clear",     {"type", "event_id"}},
            {"conversation.item.create",     {"type", "event_id", "previous_item_id", "item"}},
            {"conversation.item.retrieve",   {"type", "event_id", "item_id"}},
            {"conversation.item.delete",     {"type", "event_id", "item_id"}},
            {"conversation.item.truncate",   {"type", "event_id", "item_id", "content_index", "audio_end_ms"}},
            {"response.create",              {"type", "event_id", "response"}},
            {"response.cancel",              {"type", "event_id", "response_id"}},
        };
        const auto known = fields.find(t);
        if (known != fields.end() && !known_fields(ev, "", known->second)) {
            return;
        }
        if (t == "session.update" || t == "transcription_session.update") {
            on_session_update(ev);
        } else if (t == "input_audio_buffer.append") {
            on_append(ev);
        } else if (t == "input_audio_buffer.commit") {
            on_commit();
        } else if (t == "input_audio_buffer.clear") {
            on_clear();
        } else if (t == "conversation.item.create") {
            on_item_create(ev);
        } else if (t == "conversation.item.retrieve") {
            on_item_retrieve(ev);
        } else if (t == "conversation.item.delete") {
            on_item_delete(ev);
        } else if (t == "conversation.item.truncate") {
            on_item_truncate(ev);
        } else if (t == "response.create") {
            on_response_create(ev);
        } else if (t == "response.cancel") {
            if (!responding) {
                send_error("response.cancel: no active response", "invalid_request_error", "response_cancel_not_active");
                return;
            }
            cancel_voice();
            cancel_by_turn = false;
            cancel_sent    = true;
            control(SERVER_CONTROL_ACTION_RESPONSE_CANCEL);
        } else if (t.empty()) {
            send_error("missing event type", "invalid_request_error", "missing_required_parameter");
        } else {
            send_error("unknown event type: " + t, "invalid_request_error", "invalid_value");
        }
    }

    //
    // results of the slot
    //

    json response_json(const std::string & status, const json & status_details, const json & usage) const {
        json r = {{"object", "realtime.response"}, {"id", response_id}, {"status", status}, {"status_details", status_details},
                  {"output", output}};
        for (const auto & [k, v] : response_meta.items()) {
            r[k] = v;
        }
        r["usage"] = usage;
        return r;
    }

    // an item of the response is also an item of the conversation
    void add_output_item(int idx) {
        send_event({{"type", "response.output_item.added"}, {"response_id", response_id}, {"output_index", idx},
                    {"item", output[idx]}});
        if (resp_oob) {
            return;
        }
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", output[idx]}});
        last_item_id = output[idx]["id"];
    }

    // the message item of the response is added with its first delta
    void open_message() {
        if (idx_message >= 0) {
            return;
        }
        idx_message = (int) output.size();
        output.push_back({{"id", response_item_id}, {"type", "message"}, {"status", "in_progress"}, {"role", "assistant"},
                          {"content", json::array()}});
        add_output_item(idx_message);
    }

    // a part of the message is opened on its first delta, in order of appearance
    int open_part(bool audio) {
        open_message();
        const json part = audio ? json{{"type", "audio"}, {"transcript", ""}} : json{{"type", "text"}, {"text", ""}};
        parts.push_back(part);
        send_event({{"type", "response.content_part.added"}, {"response_id", response_id}, {"item_id", response_item_id},
                    {"output_index", idx_message}, {"content_index", parts.size() - 1}, {"part", part}});
        return (int) parts.size() - 1;
    }

    json part_base(int idx) const {
        return {{"response_id", response_id}, {"item_id", response_item_id}, {"output_index", idx_message}, {"content_index", idx}};
    }

    // a voice speaks the transcript: its text up to `end` goes out when the audio of its piece starts (mutex held)
    void send_transcript(size_t end) {
        if (!transcript_mode || end <= voice_sent) {
            return;
        }
        end = std::min(end, response_transcript.size());
        if (idx_audio < 0) {
            idx_audio = open_part(true);
        }
        json ev = part_base(idx_audio);
        ev["type"]  = "response.output_audio_transcript.delta";
        ev["delta"] = response_transcript.substr(voice_sent, end - voice_sent);
        voice_sent = end;
        send_event(ev);
    }

    void cancel_voice() {
        std::lock_guard<std::mutex> lock(mutex);
        if (voice_cancel) {
            voice_cancel->store(true);
        }
    }

    // a piece of the response text goes to the voice model; its audio comes back on the worker thread
    void speak(const server_voice_segmenter::piece & p) {
        voice_segment seg;
        seg.t0 = voice_text;
        seg.t1 = voice_text + p.raw_len;
        voice_text = seg.t1;
        if (p.text.empty()) {
            return;
        }
        const size_t idx = voice_segments.size();
        voice_segments.push_back(seg);
        voice_pending++;
        voice_jobs++;
        server_voice_job job;
        job.text      = p.text;
        job.voice     = voice_name;
        job.cancelled = voice_cancel;
        const std::string rid = response_id;
        job.on_audio = [this, rid, idx](const float * pcm, size_t n) {
            std::lock_guard<std::mutex> lock(mutex);
            if (rid != response_id || voice_cancel->load()) {
                return;
            }
            if (voice_segments[idx].s0 < 0) {
                voice_segments[idx].s0 = out_samples;
                send_transcript(voice_segments[idx].t1);
            }
            on_audio(out_rs.process(pcm, n));
            voice_segments[idx].s1 = out_samples;
        };
        job.on_done = [this, rid](bool, int32_t n_frames) {
            std::lock_guard<std::mutex> lock(mutex);
            voice_jobs--;
            if (rid != response_id) {
                return;
            }
            voice_pending--;
            voice_frames += n_frames;
            if (llm_done && voice_pending == 0 && held_done) {
                const server_task_result_realtime res = *held_done;
                held_done.reset();
                finish_response(res);
            }
        };
        deps.voices->submit(voice_model, std::move(job));
    }

    void on_text(const std::string & text) {
        if (voice_resp) {
            for (const auto & p : segmenter.push(text)) {
                speak(p);
            }
        }
        if (transcript_mode) {
            if (idx_audio < 0) {
                idx_audio = open_part(true);
            }
            response_transcript += text;
            if (voice_resp) {
                return; // it goes out with the audio, see send_transcript()
            }
            json ev = part_base(idx_audio);
            ev["type"]  = "response.output_audio_transcript.delta";
            ev["delta"] = text;
            send_event(ev);
            return;
        }
        if (idx_text < 0) {
            idx_text = open_part(false);
        }
        response_text += text;
        json ev = part_base(idx_text);
        ev["type"]  = "response.output_text.delta";
        ev["delta"] = text;
        send_event(ev);
    }

    void on_audio(const std::vector<float> & samples) {
        if (samples.empty()) {
            return;
        }
        if (idx_audio < 0) {
            idx_audio = open_part(true);
        }
        out_samples += (int64_t) samples.size();
        const std::string bytes = encode_audio(samples, resp_fmt);
        out_audio += bytes;
        json ev = part_base(idx_audio);
        ev["type"]  = "response.output_audio.delta";
        ev["delta"] = base64::encode(bytes);
        send_event(ev);
    }

    // a tool call is an output item of its own: its name comes first, then its arguments
    void on_tool_call(const common_chat_msg_diff & diff) {
        auto it = call_items.find(diff.tool_call_index);
        if (it == call_items.end()) {
            const int idx = (int) output.size();
            it = call_items.emplace(diff.tool_call_index, idx).first;
            output.push_back({{"id", "item_" + random_string()}, {"type", "function_call"}, {"status", "in_progress"},
                              {"name", diff.tool_call_delta.name}, {"call_id", diff.tool_call_delta.id}, {"arguments", ""}});
            add_output_item(idx);
        }
        if (diff.tool_call_delta.arguments.empty()) {
            return;
        }
        json & item = output[it->second];
        item["arguments"] = item["arguments"].get<std::string>() + diff.tool_call_delta.arguments;
        send_event({{"type", "response.function_call_arguments.delta"}, {"response_id", response_id}, {"item_id", item["id"]},
                    {"output_index", it->second}, {"call_id", item["call_id"]}, {"delta", diff.tool_call_delta.arguments}});
    }

    static json usage_json(const server_task_result_realtime & res) {
        const int32_t in_text = res.n_input - res.n_input_audio;
        const double  frac    = res.n_input > 0 ? (double) res.n_input_audio / res.n_input : 0.0;
        const int32_t c_audio = (int32_t) std::lround(res.n_cached * frac);
        return {
            {"total_tokens",  res.n_input + res.n_out_text + res.n_out_audio},
            {"input_tokens",  res.n_input},
            {"output_tokens", res.n_out_text + res.n_out_audio},
            {"input_token_details", {
                {"text_tokens",   in_text},
                {"audio_tokens",  res.n_input_audio},
                {"image_tokens",  0},
                {"cached_tokens", res.n_cached},
                {"cached_tokens_details", {{"text_tokens", res.n_cached - c_audio}, {"audio_tokens", c_audio}, {"image_tokens", 0}}},
            }},
            {"output_token_details", {{"text_tokens", res.n_out_text}, {"audio_tokens", res.n_out_audio}}},
        };
    }

    void on_result(const server_task_result_realtime & res) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!transcribe_item_id.empty()) {
            if (!res.content.empty()) {
                transcript += res.content.text;
                send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"delta", res.content.text}});
            }
            if (res.response_done) {
                send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"transcript", transcript},
                            {"usage", transcription_usage(res.n_input, res.n_out_text)}});
                transcribe_item_id.clear();
            }
            return;
        }

        // the text goes through the parser: content and tool calls, the reasoning of the model stays out
        std::vector<common_chat_msg_diff> diffs;
        if (parse) {
            if (!res.content.empty() || res.response_done) {
                parse->update_chat_msg(res.content, !res.response_done, diffs, true);
            }
        } else if (!res.content.empty()) {
            diffs.emplace_back();
            diffs.back().content_delta = res.content.text;
        }
        for (const auto & diff : diffs) {
            if (!diff.content_delta.empty()) {
                on_text(diff.content_delta);
            }
            if (diff.tool_call_index != std::string::npos) {
                on_tool_call(diff);
            }
        }
        if (!res.audio.empty()) {
            on_audio(out_rs.process(res.audio.data(), res.audio.size()));
        }
        if (!res.response_done) {
            return;
        }
        if (voice_resp) {
            // the model is done, the voice may still speak: the response ends with the audio of the last piece
            speak(segmenter.flush());
            llm_done = true;
            if (voice_pending > 0) {
                held_done = res;
                return;
            }
        }
        finish_response(res);
    }

    // the end of a response: done events, the items of the conversation, response.done (mutex held)
    void finish_response(const server_task_result_realtime & res_in) {
        set_hot(false);
        server_task_result_realtime res = res_in;
        res.n_out_audio += voice_frames;
        if (voice_resp && voice_cancel && voice_cancel->load()) {
            res.cancelled = true;
        }
        if (idx_audio >= 0) {
            on_audio(out_rs.flush());
        }
        if (voice_resp && !(voice_cancel && voice_cancel->load())) {
            send_transcript(response_transcript.size()); // the text no piece spoke
        }

        std::string status       = "completed";
        json        details      = nullptr;
        if (res.fault) {
            status  = "failed";
            details = {{"type", "failed"}, {"error", {{"type", res.fault_full ? "invalid_request_error" : "server_error"},
                                                      {"code", res.fault_full ? json("context_length_exceeded") : json(nullptr)},
                                                      {"message", res.fault_message}}}};
        } else if (res.cancelled) {
            status  = "cancelled";
            details = {{"type", "cancelled"}, {"reason", cancel_by_turn ? "turn_detected" : "client_cancelled"}};
        } else if (res.truncated) {
            status  = "incomplete";
            details = {{"type", "incomplete"}, {"reason", "max_output_tokens"}};
        }
        if (idx_message >= 0) {
            for (size_t i = 0; i < parts.size(); i++) {
                json ev = part_base((int) i);
                json content;
                if ((int) i == idx_text) {
                    parts[i]["text"] = response_text;
                    ev["type"] = "response.output_text.done";
                    ev["text"] = response_text;
                    send_event(ev);
                    ev.erase("text");
                    content = {{"type", "output_text"}, {"text", response_text}};
                } else {
                    parts[i]["transcript"] = response_transcript;
                    ev["type"] = "response.output_audio.done";
                    send_event(ev);
                    ev["type"]       = "response.output_audio_transcript.done";
                    ev["transcript"] = response_transcript;
                    send_event(ev);
                    ev.erase("transcript");
                    content = {{"type", "output_audio"}, {"transcript", response_transcript}};
                }
                ev["type"] = "response.content_part.done";
                ev["part"] = parts[i];
                send_event(ev);
                output[idx_message]["content"].push_back(content);
            }
        }
        for (const auto & [index, idx] : call_items) {
            json & item = output[idx];
            send_event({{"type", "response.function_call_arguments.done"}, {"response_id", response_id}, {"item_id", item["id"]},
                        {"output_index", idx}, {"call_id", item["call_id"]}, {"name", item["name"]}, {"arguments", item["arguments"]}});
        }
        for (size_t i = 0; i < output.size(); i++) {
            output[i]["status"] = status == "completed" ? "completed" : "incomplete";
            // as OpenAI: an out-of-band item gets conversation.item.done (no added) and stays out of the conversation
            send_event({{"type", "conversation.item.done"}, {"previous_item_id", i > 0 ? output[i - 1]["id"] : json(nullptr)},
                        {"item", output[i]}});
            send_event({{"type", "response.output_item.done"}, {"response_id", response_id}, {"output_index", i}, {"item", output[i]}});
            if (resp_oob) {
                continue;
            }
            // the response is in the conversation from its mark, its items go in together when it is rebuilt
            conv_item it;
            it.kind        = conv_item::ASSISTANT;
            it.id          = output[i]["id"];
            it.mark        = i == 0 ? resp_mark : "";
            it.before      = resp_before;
            it.item        = output[i];
            it.response_id = response_id;
            if ((int) i == idx_message) {
                for (const auto & seg : voice_segments) {
                    if (seg.s1 > 0) {
                        it.spoken.emplace_back(seg.s1, (int64_t) seg.t1);
                    }
                }
                it.audio_samples = out_samples;
                it.audio_rate    = resp_fmt.rate;
                it.audio         = out_audio;
                it.audio_fmt     = resp_fmt;
            }
            // items the client created during the response come after it
            const size_t pos = std::min(resp_item_pos++, items.size());
            items.insert(items.begin() + pos, std::move(it));
        }
        send_event({{"type", "response.done"}, {"response", response_json(status, details, usage_json(res))}});
        responding     = false;
        response_ended = true;
        voice_resp     = false;
        if (!res.fault) {
            n_faults = 0;
        }
        // the context window of the conversation stands in for the token budget
        send_event({{"type", "rate_limits.updated"}, {"rate_limits", json::array({{
            {"name", "tokens"}, {"limit", res.n_ctx}, {"remaining", std::max(0, res.n_ctx - res.n_past)}, {"reset_seconds", 0},
        }})}});
        parse.reset();
    }

    //
    // slot errors and truncation
    //

    // about the tokens of the instructions and tools: the text before the first user turn
    int32_t n_instructions() const {
        return (int32_t) common_tokenize(deps.vocab, first_header, false, true).size();
    }

    // the size the conversation can have when a response starts, 0: all of the context
    // a quarter of the context stays free for the answer, or its max_output_tokens if that is less
    int32_t response_limit(int32_t n_predict) const {
        if (!trunc.on) {
            return 0;
        }
        const int32_t n_ctx = deps.meta.slot_n_ctx;
        int32_t limit = n_ctx - (n_predict > 0 ? std::min(n_predict, n_ctx / 4) : n_ctx / 4);
        if (trunc.limit > 0) {
            limit = std::min(limit, n_instructions() + trunc.limit);
        }
        return limit;
    }

    // truncation: the oldest turns leave the context until the rest takes the retention ratio of the limit, one turn at
    // least; the client keeps their items. The sizes come from the marks of the slot, without them a share of the turns goes.
    // keep_last: the last turn stays. Returns the number of items that left
    size_t truncate_items(const server_task_result_realtime & f, bool keep_last) {
        std::lock_guard<std::mutex> lock(mutex);
        // a turn starts at a message or at the first item of a response; tool results stay with their response
        std::vector<size_t> starts;
        for (size_t i = 0; i < items.size(); i++) {
            const bool same = i > 0 && items[i].kind == conv_item::ASSISTANT && items[i - 1].kind == conv_item::ASSISTANT &&
                              items[i].response_id == items[i - 1].response_id;
            if (i == 0 || (!same && items[i].kind != conv_item::TOOL_OUTPUT)) {
                starts.push_back(i);
            }
        }
        const size_t n_max = starts.size() - (keep_last && !starts.empty() ? 1 : 0);
        if (n_max == 0) {
            return 0;
        }
        auto pos_of = [&](size_t turn) {
            for (const auto & m : f.marks) {
                if (m.first == items[starts[turn]].mark) {
                    return m.second;
                }
            }
            return -1;
        };
        const int32_t full   = trunc.limit > 0 ? std::min(f.n_ctx, n_instructions() + trunc.limit) : f.n_ctx;
        const int32_t target = std::min((int32_t) (trunc.ratio * full), f.n_ctx - f.n_ctx / 4);
        size_t n = 1;
        if (pos_of(0) < 0) {
            // the slot holds another context (a response with its own)
            n = starts.size() - (size_t) (trunc.ratio * starts.size());
        } else {
            const int32_t n_sys = n_instructions();
            for (; n < n_max; n++) {
                // a turn the slot did not get to has no mark: what stays is as good as empty
                const int32_t pos = pos_of(n);
                if (pos < 0 || n_sys + f.n_past - pos <= target) {
                    break;
                }
            }
        }
        n = std::max<size_t>(1, std::min(n, n_max));
        // the context does not start with an answer if a user turn can come first
        while (n < n_max && items[starts[n]].kind == conv_item::ASSISTANT) {
            n++;
        }
        const size_t n_items = n < starts.size() ? starts[n] : items.size();
        old_items.insert(old_items.end(), std::make_move_iterator(items.begin()), std::make_move_iterator(items.begin() + n_items));
        items.erase(items.begin(), items.begin() + n_items);
        return n_items;
    }

    // back to the context of the session if a response with its own was in the slot; the snapshots go
    void reset_context() {
        if (pending_restore) {
            pending_restore = false;
            snap_restore    = false;
            instructions = saved_instructions;
            tools        = saved_tools;
            tools_json   = saved_tools_json;
            configure();
        }
        own_ctx      = {};
        session_snap = {};
        control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, "session");
        for (const auto & snap : snapshots) {
            control(SERVER_CONTROL_ACTION_SNAPSHOT_DROP, snap.name);
        }
        snapshots.clear();
    }

    // the result that ends the response in progress after a slot error (mutex held)
    server_task_result_realtime failed_result(const server_task_result_realtime & f) {
        server_task_result_realtime res = held_done ? *held_done : server_task_result_realtime();
        held_done.reset();
        res.response_done = true;
        res.fault         = true;
        res.fault_full    = f.fault_full;
        res.fault_message = f.fault_message;
        res.n_ctx         = f.n_ctx;
        res.n_past        = f.n_past;
        return res;
    }

    // a control of the slot failed (result loop)
    void on_fault(const server_task_result_realtime & f) {
        fault_pending = true;
        std::lock_guard<std::mutex> conv(conv_mutex);
        fault_pending = false;
        SRV_WRN("realtime session %s: %s\n", id.c_str(), f.fault_message.c_str());
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!transcribe_item_id.empty()) {
                send_event({{"type", "conversation.item.input_audio_transcription.failed"}, {"item_id", transcribe_item_id}, {"content_index", 0},
                            {"error", {{"type", "transcription_error"}, {"code", nullptr}, {"message", f.fault_message}, {"param", nullptr}}}});
                transcribe_item_id.clear();
            }
        }
        if (f.n_clears != n_clears) {
            return; // a CLEAR sent since then starts the conversation again
        }
        if (type == "transcription") {
            // each buffer is transcribed on its own, the one in progress is lost
            control(SERVER_CONTROL_ACTION_CLEAR);
            load_turn({});
            reset_audio();
            send_error_event(f.fault_message, f.fault_full ? "invalid_request_error" : "server_error",
                             f.fault_full ? "context_length_exceeded" : "", "");
            send_event({{"type", "input_audio_buffer.cleared"}});
            return;
        }
        recover(f);
    }

    // the slot runs nothing until the next CLEAR: the response in progress starts again or fails, old items leave a
    // context that was full, then the conversation goes into the slot again (conv_mutex held)
    void recover(const server_task_result_realtime & f) {
        bool lost   = f.fault_full ? !trunc.on : ++n_faults > 1; // no way to put the conversation in again
        bool report = !f.fault_full || lost;                     // with an error event, unless a response fails with the error
        bool retry  = false;
        if (responding) {
            cancel_voice();
            std::lock_guard<std::mutex> lock(mutex);
            if (cancel_sent) {
                server_task_result_realtime res = failed_result(f);
                res.fault     = false;
                res.cancelled = true;
                finish_response(res);
            } else if (output.empty() && !llm_done && !lost && (f.fault_full || resp_retries++ == 0)) {
                retry = true; // the client got nothing of it yet
            } else {
                finish_response(failed_result(f));
                report = false;
            }
        }
        if (f.fault_full && trunc.on && truncate_items(f, true) == 0) {
            // the last turn is all the context holds
            if (retry && !resp_tight) {
                resp_tight = true;
            } else if (audio_open && !pending_pcm.empty()) {
                reset_audio();
                send_error_event("the audio of the input audio buffer does not fit in the context, the buffer is cleared",
                                 "invalid_request_error", "context_length_exceeded", "");
                send_event({{"type", "input_audio_buffer.cleared"}});
            } else {
                // the turn does not fit on its own: it leaves the context without an answer
                retry  = false;
                report = true;
                lost   = truncate_items(f, false) == 0;
            }
        }
        if (responding && !retry) {
            std::lock_guard<std::mutex> lock(mutex);
            finish_response(failed_result(f));
            report = false;
        }
        if (report) {
            send_error_event(f.fault_message, f.fault_full ? "invalid_request_error" : "server_error",
                             f.fault_full ? "context_length_exceeded" : "", "");
        }
        reset_context();
        if (lost) {
            ctx_lost = true; // the next response tries again
            if (!holding()) {
                run_held();
            }
            return;
        }
        audio_buf held = take_audio();
        put_conversation();
        if (retry) {
            responding = false;
            if (!on_response_create(resp_event, resp_from_client, true)) {
                std::lock_guard<std::mutex> lock(mutex);
                finish_response(failed_result(f));
            }
        }
        reopen_audio(std::move(held));
        if (!holding()) {
            run_held();
        }
    }

    //
    // session
    //

    // an error of the task ends the session
    void send_slot_error(const json & err) {
        const bool full = json_value(err, "type", std::string()) == "exceed_context_size_error";
        send_error_event(json_value(err, "message", std::string("server error")), full ? "invalid_request_error" : "server_error",
                         full ? "context_length_exceeded" : "", "");
    }

    void run() {
        model_in_rate = cascade() ? deps.asr->sample_rate() : deps.mctx ? mtmd_get_audio_sample_rate(deps.mctx) : 16000;
        if (deps.cap.audio_gen) {
            model_out_rate = mtmd_gen_audio_get_info(deps.mctx).sample_rate;
        }
        in_rs.reset(in_fmt.rate, model_in_rate);
        out_rs.reset(model_out_rate, out_fmt.rate);
        if (!configure()) {
            send_error("the chat template does not fit the realtime turn format", "server_error");
            conn.close();
            return;
        }

        server_response_reader rd(deps.queue_tasks, deps.queue_results, 1);
        {
            server_task task(SERVER_TASK_TYPE_REALTIME);
            task.id                       = rd.get_new_id();
            task.params.sampling          = deps.params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = id;
            task.params.oaicompat_model   = deps.meta.model_name;
            rd.post_task(std::move(task));
        }
        // the slot answers once it holds the session
        {
            auto result = rd.next([this]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_slot_error(result->to_json());
                }
                conn.close();
                return;
            }
        }
        send_event({{"type", "session.created"}, {"session", session_json()}});

        // client events are handled on this thread, results are sent from the loop below
        reset_detector();
        std::thread reader([this]() {
            std::string msg;
            while (conn.read(msg) != 0) {
                std::lock_guard<std::mutex> lock(conv_mutex);
                try {
                    handle(msg);
                } catch (const std::exception & e) {
                    send_error(e.what(), "server_error");
                }
            }
            closed = true;
            conn.close();
        });

        while (true) {
            auto result = rd.next([this]() { return !conn.is_open(); });
            if (!result) {
                break;
            }
            if (result->is_error()) {
                send_slot_error(result->to_json());
                break;
            }
            if (auto * res = dynamic_cast<server_task_result_realtime *>(result.get())) {
                try {
                    if (res->fault) {
                        on_fault(*res);
                    } else {
                        on_result(*res);
                    }
                } catch (const std::exception & e) {
                    send_error_event(e.what(), "server_error", "", "");
                    break;
                }
            }
        }

        closed = true;
        conn.close();
        reader.join();
        set_hot(false);
        if (asr_turn) {
            deps.asr->cancel(asr_turn); // its sequence of the ASR model is free again
            asr_turn.reset();
        }
        for (auto & t : transcribers) {
            t.join();
        }
        // the voice jobs call back into the session: stop them and wait until each one is done
        cancel_voice();
        while (true) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (voice_jobs == 0) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

} // namespace

void server_realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn) {
    realtime_conversation session(deps, conn);
    session.run();
}

//
// server_realtime_slot
//

server_realtime_slot::server_realtime_slot(llama_context * ctx, mtmd_context * mctx, llama_seq_id seq_id, int32_t n_ctx,
                                           const common_params_sampling & sampling, server_response & queue_results, int id_task, int index,
                                           bool swa_full, server_realtime_prompts & prompts)
    : ctx(ctx), model(llama_get_model(ctx)), vocab(llama_model_get_vocab(llama_get_model(ctx))), mctx(mctx), seq_id(seq_id),
      n_ctx(n_ctx), sampling(sampling), queue_results(queue_results), id_task(id_task), index(index), swa_full(swa_full),
      prompts(prompts) {
    if (mctx && mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE) {
        gen.init(ctx, mctx);
    }
    streaming = mctx && mtmd_support_audio_stream(mctx);
}

void server_realtime_slot::push(server_realtime_op && op) {
    ops.push_back(std::move(op));
}

void server_realtime_slot::cancel() {
    cancelled = responding;
}

bool server_realtime_slot::is_idle() const {
    return ops.empty() && !responding;
}

bool server_realtime_slot::fail(const std::string & msg, enum error_type type) {
    error      = msg;
    error_type = type;
    return false;
}

void server_realtime_slot::send_result(common_chat_input content, std::vector<float> audio, bool response_done) {
    auto res = std::make_unique<server_task_result_realtime>();
    res->id            = id_task;
    res->index         = index;
    res->content       = std::move(content);
    res->audio         = std::move(audio);
    res->response_done = response_done;
    res->cancelled     = response_done && cancelled;
    res->n_decoded     = n_decoded;
    if (response_done) {
        res->truncated     = truncated;
        res->n_input       = n_input;
        res->n_input_audio = n_input_audio;
        res->n_cached      = std::min(n_prev_end, n_input);
        res->n_out_text    = n_out_text;
        res->n_out_audio   = n_out_audio;
        res->n_ctx         = n_ctx;
        res->n_past        = n_past;
    }
    queue_results.send(std::move(res));
}

// appends tokens or embedding rows to the conversation, logits of the last one if asked
bool server_realtime_slot::eval(const llama_token * tokens, const float * rows, int32_t n, llama_pos p0, bool logits_last) {
    const int32_t n_embd  = llama_model_n_embd_inp(model);
    const int32_t n_batch = llama_n_batch(ctx);
    for (int32_t i0 = 0; i0 < n; i0 += n_batch) {
        const int32_t n_cur = std::min(n_batch, n - i0);
        llama_batch b = llama_batch_init(n_cur, rows ? n_embd : 0, 1);
        for (int32_t i = 0; i < n_cur; i++) {
            if (rows) {
                memcpy(b.embd + (size_t) i * n_embd, rows + (size_t) (i0 + i) * n_embd, n_embd * sizeof(float));
            } else {
                b.token[i] = tokens[i0 + i];
            }
            b.pos[i]       = p0 + i0 + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seq_id;
            b.logits[i]    = logits_last && i0 + i == n - 1;
        }
        b.n_tokens = n_cur;
        const int ret = llama_decode(ctx, b);
        llama_batch_free(b);
        if (ret != 0) {
            return fail("failed to decode the conversation");
        }
    }
    return true;
}

// tokens wait for the next decode that needs them in the memory, rows go in at once
bool server_realtime_slot::decode(const llama_token * tokens, const float * rows, int32_t n, bool logits_last) {
    if (n_past + n >= n_ctx) {
        return fail("the conversation does not fit in the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
    }
    if (tokens) {
        pending.insert(pending.end(), tokens, tokens + n);
        n_past += n;
        return !logits_last || flush(true);
    }
    if (!flush() || !eval(nullptr, rows, n, n_past, logits_last)) {
        return false;
    }
    n_past  += n;
    n_audio += n;
    return true;
}

bool server_realtime_slot::flush(bool logits_last) {
    const llama_tokens toks = std::move(pending);
    pending.clear();
    if (toks.empty()) {
        return true;
    }
    const llama_pos p0 = n_past - (llama_pos) toks.size();
    // the start of a context: a long prompt may be in the memory of an earlier context already
    // the state is cut to the common part, with a sliding window that needs the full cache (--swa-full)
    const bool    start  = p0 == 0 && toks.size() >= server_realtime_prompts::min_tokens && (swa_full || llama_model_n_swa(model) == 0);
    const int32_t n_have = start ? load_prompt(toks) : 0;
    if (!eval(toks.data() + n_have, nullptr, (int32_t) toks.size() - n_have, p0 + n_have, logits_last)) {
        return false;
    }
    if (start && n_have == 0) {
        save_prompt(toks);
    }
    return true;
}

// the positions that the state of an earlier prompt fills, 0 if none starts like toks
int32_t server_realtime_slot::load_prompt(const llama_tokens & toks) {
    server_realtime_prompts::entry * best = nullptr;
    size_t n_best = 0;
    for (auto & e : prompts.entries) {
        size_t n = 0;
        while (n < e.tokens.size() && n < toks.size() && e.tokens[n] == toks[n]) {
            n++;
        }
        if (n > n_best) {
            best   = &e;
            n_best = n;
        }
    }
    n_best = std::min(n_best, toks.size() - 1); // the last token is decoded for its logits
    if (!best || n_best < server_realtime_prompts::min_tokens) {
        return 0;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (llama_state_seq_set_data_ext(ctx, best->state.data(), best->state.size(), seq_id, 0) != best->state.size() ||
        !llama_memory_seq_rm(mem, seq_id, (llama_pos) n_best, -1)) {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
        return 0;
    }
    best->used = ++prompts.n_use;
    return (int32_t) n_best;
}

void server_realtime_slot::save_prompt(const llama_tokens & toks) {
    server_realtime_prompts::entry e;
    e.tokens = toks;
    e.used   = ++prompts.n_use;
    e.state.resize(llama_state_seq_get_size_ext(ctx, seq_id, 0));
    if (llama_state_seq_get_data_ext(ctx, e.state.data(), e.state.size(), seq_id, 0) != e.state.size()) {
        return;
    }
    auto & all = prompts.entries;
    if (all.size() >= server_realtime_prompts::max_entries) {
        all.erase(std::min_element(all.begin(), all.end(), [](const auto & a, const auto & b) { return a.used < b.used; }));
    }
    all.push_back(std::move(e));
}

// decodes the rows the audio stream has ready
bool server_realtime_slot::audio_rows() {
    while (true) {
        const int32_t n = mtmd_audio_stream_step(stream.get(), LLAMA_TOKEN_NULL);
        if (n == -2) {
            return fail("audio stream error");
        }
        if (n <= 0) {
            return true;
        }
        const llama_token * tokens = mtmd_audio_stream_get_output_tokens(stream.get());
        const float *       rows   = tokens ? nullptr : mtmd_audio_stream_get_output_embd(stream.get());
        if (!decode(tokens, rows, n, false)) {
            return false;
        }
        // the chunks of the stream are kept as one part per run of tokens or rows: it goes in again in one decode
        auto & parts = audio_open.parts;
        if (parts.empty() || parts.back().tokens.empty() != !tokens) {
            parts.emplace_back();
        }
        if (tokens) {
            parts.back().tokens.insert(parts.back().tokens.end(), tokens, tokens + n);
        } else {
            parts.back().rows.insert(parts.back().rows.end(), rows, rows + (size_t) n * llama_model_n_embd_inp(model));
        }
    }
}

// the end of the audio of a turn; pcm: the whole audio of a turn that goes in again (not of the stream)
bool server_realtime_slot::audio_end(const std::vector<float> & pcm) {
    if (!streaming) {
        turn_pcm.insert(turn_pcm.end(), pcm.begin(), pcm.end());
        return audio_whole();
    }
    if (!stream && !pcm.empty()) {
        for (const auto & turn : audio_turns) {
            if (turn.pcm != pcm) {
                continue;
            }
            for (const auto & part : turn.parts) {
                const bool    rows = part.tokens.empty();
                const int32_t n    = rows ? (int32_t) (part.rows.size() / llama_model_n_embd_inp(model)) : (int32_t) part.tokens.size();
                if (!decode(rows ? nullptr : part.tokens.data(), rows ? part.rows.data() : nullptr, n, false)) {
                    return false;
                }
            }
            return true;
        }
        stream.reset(mtmd_audio_stream_init(mctx, mtmd_audio_stream_params_default()));
        if (!stream) {
            return fail("failed to initialize the audio stream");
        }
    }
    if (!stream) {
        return true;
    }
    if (!pcm.empty()) {
        audio_open.pcm.insert(audio_open.pcm.end(), pcm.begin(), pcm.end());
        mtmd_audio_stream_push(stream.get(), pcm.data(), pcm.size());
    }
    mtmd_audio_stream_flush(stream.get(), true);
    if (!audio_rows()) {
        return false;
    }
    stream.reset();
    if (audio_turns.size() >= max_audio_turns) {
        audio_turns.pop_front();
    }
    audio_turns.push_back(std::move(audio_open));
    audio_open = {};
    return true;
}

// encodes the audio of a turn in one go, for models without an audio stream: the media chunks of mtmd carry the
// model's own audio markers and split long audio as its encoder needs
bool server_realtime_slot::audio_whole() {
    std::vector<float> pcm = std::move(turn_pcm);
    turn_pcm.clear();
    if (pcm.empty()) {
        return true;
    }
    mtmd::bitmap bmp(mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()));
    if (!bmp.ptr) {
        return fail("failed to load the audio of the turn");
    }
    mtmd::input_chunks    chunks(mtmd_input_chunks_init());
    const mtmd_input_part part     = { nullptr, bmp.ptr.get() };
    const mtmd_input_part * parts[] = { &part };
    if (mtmd_tokenize_from_parts(mctx, chunks.ptr.get(), parts, 1, false) != 0) {
        return fail("failed to tokenize the audio of the turn");
    }
    if (n_past + mtmd_helper_get_n_pos(chunks.ptr.get()) >= n_ctx) {
        return fail("the conversation does not fit in the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
    }
    int32_t n_rows = 0;
    for (size_t i = 0; i < chunks.size(); i++) {
        if (mtmd_input_chunk_get_type(chunks[i]) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            n_rows += (int32_t) mtmd_input_chunk_get_n_tokens(chunks[i]);
        }
    }
    if (!flush()) {
        return false;
    }
    llama_pos new_n_past = n_past;
    if (mtmd_helper_eval_chunks(mctx, ctx, chunks.ptr.get(), n_past, seq_id, llama_n_batch(ctx), false, &new_n_past) != 0) {
        return fail("failed to encode the audio of the turn");
    }
    n_past   = new_n_past;
    n_audio += n_rows;
    return true;
}

// samples the next text token of a response from the logits of the slot's own last decode; several slots
// respond in turns on the same context, so this runs before any other slot decodes
void server_realtime_slot::sample_next() {
    sampled = LLAMA_TOKEN_NULL;
    if (gen.valid() && !gen.next_is_text()) {
        return;
    }
    sampled = common_sampler_sample(smpl.get(), ctx, -1);
    common_sampler_accept(smpl.get(), sampled, true);
}

bool server_realtime_slot::response_start(const server_realtime_op & op) {
    auto sparams = op.response.sampling;
    n_predict   = op.response.n_predict;
    n_decoded   = 0;
    n_out_text  = 0;
    n_out_audio = 0;
    cancelled   = false;
    truncated   = false;

    // the answer needs room: with a limit the session takes old items out of the conversation and asks again
    const llama_tokens toks = common_tokenize(vocab, op.text, false, true);
    const int32_t      n_in = n_past + (int32_t) toks.size();
    if (n_in + 8 >= n_ctx || (op.response.n_limit > 0 && n_in > op.response.n_limit)) {
        return fail("the conversation does not leave room for the response in the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
    }

    if (gen.valid()) {
        if (!flush()) {
            return false;
        }
        // the gen-audio pipeline continues the conversation and follows the model's text/audio switching
        mtmd_helper::gen_audio::inp inp;
        inp.set_prompt(op.text);
        inp.data.seq_id   = seq_id;
        inp.data.n_past   = n_past;
        inp.data.seed     = sampling.seed;
        inp.data.stream   = true;
        inp.data.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        inp.data.out_mode = op.response.out_mode;
        inp.set_lang(op.response.lang);
        // predefined voice -> speaker ref, LFM3 uses the system prompt instead
        if (!op.response.voice.empty() && mtmd_gen_audio_get_info(mctx).n_voices > 0) {
            mtmd::bitmap_ptr voice(mtmd_gen_audio_get_voice(mctx, op.response.voice.c_str()));
            if (!voice) {
                return fail("unknown voice \"" + op.response.voice + "\"", ERROR_TYPE_INVALID_REQUEST);
            }
            inp.set_speaker_ref(std::move(voice));
        }
        if (gen.set_input(inp.get()) != 0) {
            return fail("failed to start the response");
        }
        // the pipeline may bias some text tokens, e.g. the switch to audio of a text-only response
        for (const auto & b : gen.logit_bias()) {
            sparams.logit_bias.push_back(b);
        }
        int32_t ret;
        while ((ret = gen.step_prompt(llama_n_batch(ctx))) > 0) {
        }
        if (ret < 0) {
            return fail("failed to start the response");
        }
        const float * h = llama_get_embeddings_ith(ctx, -1);
        h_state.assign(h, h + llama_model_n_embd(model));
    } else {
        if (toks.empty()) {
            return fail("failed to start the response");
        }
        if (!decode(toks.data(), nullptr, (int32_t) toks.size(), true)) {
            return false;
        }
    }
    n_input       = gen.valid() ? llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1 : n_past;
    n_input_audio = n_audio;
    smpl.reset(common_sampler_init(model, sparams));
    preserved = sparams.preserved_tokens;
    // the logits of the last decode are this slot's only until another slot decodes
    sample_next();
    responding = true;
    return true;
}

// control tokens stay out of the text, except the preserved ones the client parses (tool calls)
std::string server_realtime_slot::piece(llama_token tok) const {
    const bool p = preserved.count(tok) > 0;
    return p || !llama_vocab_is_control(vocab, tok) ? common_token_to_piece(vocab, tok, p) : "";
}

// one text token or audio frame of the response being generated
bool server_realtime_slot::response_step() {
    bool stop = false;
    common_chat_input content;
    std::vector<float> audio;

    if (gen.valid()) {
        const float * h_next = nullptr;
        if (gen.step_gen(sampled, h_state.data(), &h_next, &stop) != 0) {
            return fail("response generation failed");
        }
        if (sampled != LLAMA_TOKEN_NULL && !stop) {
            content.append(piece(sampled), sampled);
        }
        (sampled != LLAMA_TOKEN_NULL ? n_out_text : n_out_audio)++;
        if (h_next) {
            h_state.assign(h_next, h_next + llama_model_n_embd(model));
        }
    } else {
        stop = llama_vocab_is_eog(vocab, sampled);
        if (!stop) {
            content.append(piece(sampled), sampled);
            n_out_text++;
            if (!decode(&sampled, nullptr, 1, true)) {
                return false;
            }
        }
    }
    n_decoded++;

    // the pipeline decodes on its own, the response ends before the context is full
    const bool ctx_full = gen.valid() ?
        llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 2 >= n_ctx :
        n_past + 1 >= n_ctx;
    const bool done = stop || cancelled || ctx_full || (n_predict > 0 && n_decoded >= n_predict);
    truncated = done && !stop && !cancelled;
    if (!done) {
        sample_next();
    }
    if (gen.valid()) {
        if (done) {
            gen.flush();
        }
        int32_t sample_rate = 0;
        const char * data   = nullptr;
        size_t data_len     = 0;
        if (gen.get_output(&sample_rate, &data, &data_len) == 0 && data_len > 0) {
            audio.assign((const float *) data, (const float *) data + data_len / sizeof(float));
        }
    }
    if (!content.empty() || !audio.empty()) {
        send_result(std::move(content), std::move(audio));
    }
    if (done) {
        if (gen.valid()) {
            // the pipeline decoded the response, the sequence ends where it stopped
            n_past   = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) + 1;
            n_audio += n_out_audio; // the frames it spoke stay in the conversation
        }
        send_result({}, {}, true);
        n_prev_end = n_past;
        responding = false;
        cancelled  = false;
    }
    return true;
}

// the part of the memory that cannot drop the end of a sequence (recurrent, SWA) is saved with the mark,
// llama.cpp decides what that is; for plain attention it is empty
bool server_realtime_slot::set_mark(const std::string & name) {
    mark m;
    m.name       = name;
    m.n_past     = n_past;
    m.n_audio    = n_audio;
    m.n_prev_end = n_prev_end;
    if (n_past > 0 && !swa_full) {
        if (!flush()) {
            return false;
        }
        m.state.resize(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (llama_state_seq_get_data_ext(ctx, m.state.data(), m.state.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != m.state.size()) {
            return fail("failed to save the conversation state");
        }
    }
    marks.push_back(std::move(m));
    return true;
}

bool server_realtime_slot::rollback(const std::string & name) {
    if (name.empty()) {
        // the start of the conversation: a mark there that always exists
        marks.insert(marks.begin(), mark());
    }
    auto it = name.empty() ? marks.begin() : std::find_if(marks.begin(), marks.end(), [&](const mark & m) { return m.name == name; });
    if (it == marks.end()) {
        return fail("unknown conversation mark " + name);
    }
    llama_memory_t  mem   = llama_get_memory(ctx);
    const llama_pos n_mem = n_past - (llama_pos) pending.size();
    if (it->n_past >= n_mem) {
        // the memory is as it was at the mark
        pending.resize(it->n_past - n_mem);
    } else if (it->n_past == 0) {
        pending.clear();
        llama_memory_seq_rm(mem, seq_id, -1, -1);
    } else {
        pending.clear();
        if (!it->state.empty() &&
            llama_state_seq_set_data_ext(ctx, it->state.data(), it->state.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != it->state.size()) {
            return fail("failed to restore the conversation state");
        }
        if (!llama_memory_seq_rm(mem, seq_id, it->n_past, -1)) {
            return fail("the model memory cannot go back in the conversation");
        }
    }
    n_past     = it->n_past;
    n_audio    = it->n_audio;
    n_prev_end = it->n_prev_end;
    stream.reset();
    audio_open = {};
    turn_pcm.clear();
    marks.erase(it, marks.end());
    return true;
}

bool server_realtime_slot::save_snapshot(const std::string & name) {
    if (!flush()) {
        return false;
    }
    snapshot s;
    s.n_past     = n_past;
    s.n_audio    = n_audio;
    s.n_prev_end = n_prev_end;
    s.marks      = marks;
    if (n_past > 0) {
        s.state.resize(llama_state_seq_get_size_ext(ctx, seq_id, 0));
        if (llama_state_seq_get_data_ext(ctx, s.state.data(), s.state.size(), seq_id, 0) != s.state.size()) {
            return fail("failed to save the conversation snapshot");
        }
    }
    snapshots[name] = std::move(s);
    return true;
}

bool server_realtime_slot::load_snapshot(const std::string & name) {
    auto it = snapshots.find(name);
    if (it == snapshots.end()) {
        return fail("unknown conversation snapshot " + name);
    }
    const snapshot & s = it->second;
    pending.clear();
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    if (!s.state.empty() &&
        llama_state_seq_set_data_ext(ctx, s.state.data(), s.state.size(), seq_id, 0) != s.state.size()) {
        return fail("failed to load the conversation snapshot");
    }
    n_past     = s.n_past;
    n_audio    = s.n_audio;
    n_prev_end = s.n_prev_end;
    marks      = s.marks;
    stream.reset();
    audio_open = {};
    turn_pcm.clear();
    return true;
}

void server_realtime_slot::update() {
    // the hidden states drive the audio of a model that speaks itself; a text model needs no embeddings
    llama_set_embeddings(ctx, gen.valid());

    if (run()) {
        return;
    }
    // what the slot holds is not the conversation of the session any more
    auto res = std::make_unique<server_task_result_realtime>();
    res->id            = id_task;
    res->index         = index;
    res->fault         = true;
    res->fault_full    = error_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE;
    res->fault_message = error;
    res->n_clears      = n_clears;
    res->n_ctx         = n_ctx;
    res->n_past        = n_past;
    for (const auto & m : marks) {
        res->marks.emplace_back(m.name, m.n_past);
    }
    queue_results.send(std::move(res));
    faulted    = true;
    responding = false;
    cancelled  = false;
    stream.reset();
    turn_pcm.clear();
}

bool server_realtime_slot::run() {
    if (responding) {
        return response_step();
    }

    while (!ops.empty() && !responding) {
        server_realtime_op op = std::move(ops.front());
        ops.pop_front();
        if (faulted && op.action != SERVER_CONTROL_ACTION_CLEAR && op.action != SERVER_CONTROL_ACTION_SNAPSHOT_DROP) {
            continue;
        }
        switch (op.action) {
            case SERVER_CONTROL_ACTION_TEXT_APPEND:
                {
                    const llama_tokens toks = common_tokenize(vocab, op.text, false, true);
                    if (!toks.empty() && !decode(toks.data(), nullptr, (int32_t) toks.size(), false)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_AUDIO_APPEND:
                {
                    if (!streaming) {
                        turn_pcm.insert(turn_pcm.end(), op.audio.begin(), op.audio.end());
                        break;
                    }
                    if (!stream) {
                        stream.reset(mtmd_audio_stream_init(mctx, mtmd_audio_stream_params_default()));
                        if (!stream) {
                            return fail("failed to initialize the audio stream");
                        }
                    }
                    audio_open.pcm.insert(audio_open.pcm.end(), op.audio.begin(), op.audio.end());
                    mtmd_audio_stream_push(stream.get(), op.audio.data(), op.audio.size());
                    if (!audio_rows()) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_AUDIO_COMMIT:
            case SERVER_CONTROL_ACTION_AUDIO_END:
                {
                    if (!audio_end(op.audio)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_RESPONSE_CREATE:
                {
                    if (!response_start(op)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_CLEAR:
                {
                    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
                    stream.reset();
                    audio_open = {};
                    turn_pcm.clear();
                    marks.clear();
                    pending.clear();
                    n_past     = 0;
                    n_audio    = 0;
                    n_prev_end = 0;
                    faulted    = false;
                    n_clears++;
                } break;
            case SERVER_CONTROL_ACTION_MARK:
                {
                    if (!set_mark(op.text)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_ROLLBACK:
                {
                    if (!rollback(op.text)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_SNAPSHOT_SAVE:
                {
                    if (!save_snapshot(op.text)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_SNAPSHOT_LOAD:
                {
                    if (!load_snapshot(op.text)) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_SNAPSHOT_DROP:
                {
                    snapshots.erase(op.text);
                } break;
            default:
                break;
        }
    }
    return true;
}
