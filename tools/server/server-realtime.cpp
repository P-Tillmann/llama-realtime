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
    if (!j.is_string()) {
        err = "tool_choice must be \"auto\", \"none\" or \"required\"";
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
    mtmd_helper_gen_audio_out_mode out_mode          = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
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
    json        truncation      = "auto";
    json        tracing         = nullptr;
    json        include         = nullptr;
    json        prompt          = nullptr;

    std::optional<server_turn_params>     turn_params;     // null: the client commits the audio
    std::unique_ptr<server_turn_detector> detector;        // owned by the reader thread
    std::string                           speech_item_id;  // the user item of the speech in progress
    std::atomic<bool>                     responding{false};
    std::atomic<bool>                     cancel_sent{false};   // the response in progress was cancelled, it ends soon
    std::atomic<bool>                     cancel_by_turn{false}; // by the start of speech (turn detection)
    std::atomic<bool>                     response_ended{false}; // for the idle timeout of the detector

    pcm_resampler in_rs;  // client rate -> model rate, owned by the reader thread
    pcm_resampler out_rs; // model rate -> client rate, owned by the result loop

    // tools, in the system turn
    json                          tools_json  = json::array(); // as the client sent them
    std::vector<common_chat_tool> tools;
    common_chat_tool_choice       tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

    // the text around a turn, from the chat template
    std::string first_header; // system prompt + first user header
    std::string user_header;
    std::string user_footer;
    std::string assistant_header;
    std::string assistant_footer;
    std::string tool_header;  // around a tool result
    std::string tool_footer;
    bool        has_tool_text = false; // the chat template renders tool results

    // conversation state, owned by the reader thread
    bool        started   = false; // something went into the slot, the session type is fixed
    bool        turn_open = false; // a user turn is being filled
    bool        has_audio = false; // the open turn has audio not committed yet
    bool        tool_results = false; // tool results went in after the last response, the next one answers them
    int         n_turns   = 0;
    std::string next_header;       // text that goes in before the next user turn

    // the turn text state, kept with each item: the conversation can be rebuilt from any item
    struct turn_state {
        bool        turn_open    = false;
        bool        tool_results = false;
        int         n_turns      = 0;
        std::string next_header;
    };

    // an item of the conversation and what it takes to put it in again
    struct conv_item {
        enum kind_t { USER_TEXT, USER_AUDIO, TOOL_OUTPUT, ASSISTANT } kind;
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
    std::vector<conv_item> items; // guarded by mutex
    int                    n_marks = 0;

    // the item being filled: its mark and turn state, set before its first content
    std::string pending_mark;
    turn_state  pending_before;
    bool        audio_open = false;    // the input audio buffer has an item mark
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

    realtime_conversation(const server_realtime_deps & deps, server_http_context::ws_conn & conn) : deps(deps), conn(conn) {}

    //
    // events
    //

    void send_event(json ev) {
        ev["event_id"] = "event_" + random_string();
        conn.send(safe_json_to_str(ev));
    }

    void send_error(const std::string & message, const std::string & type_err = "invalid_request_error") {
        send_event({
            {"type",  "error"},
            {"error", {{"type", type_err}, {"code", nullptr}, {"message", message}, {"param", nullptr}}},
        });
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
            {"tool_choice",       tool_choice_str(tool_choice)},
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
        ctl.post_task(std::move(task));
        auto result = ctl.next([this]() { return closed.load() || !conn.is_open(); });
        auto * res = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
        return res && res->success;
    }

    // sampling of a response: the server defaults, with the tool call grammar of chat
    common_params_sampling response_sampling(const common_chat_session * chat) const {
        common_params_sampling sampling = deps.sampling;
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
            const std::string g = render_with_instructions({user}, true);
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
        assistant_footer = c.substr(pc + reply.size());
        user_header      = d.substr(c.size(), pd - c.size());

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
        } catch (const std::exception &) {
            // the template has no tool messages
        }
        tools = session_tools;
        return has_tool_text || session_tools.empty();
    }

    //
    // conversation
    //

    // the header of a user turn goes in before its first audio or text
    bool ensure_turn() {
        if (turn_open) {
            return true;
        }
        const std::string header = n_turns == 0 ? first_header : next_header + user_header;
        if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, header)) {
            return false;
        }
        next_header.clear();
        started   = true;
        turn_open = true;
        return true;
    }

    turn_state save_turn() const {
        return {turn_open, tool_results, n_turns, next_header};
    }

    void load_turn(const turn_state & t) {
        turn_open    = t.turn_open;
        tool_results = t.tool_results;
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
                if (!ensure_turn() || (!it.pcm.empty() && !control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, it.pcm))) {
                    return false;
                }
                return control(SERVER_CONTROL_ACTION_AUDIO_END);
            case conv_item::TOOL_OUTPUT:
                if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, next_header + tool_header + it.text + tool_footer)) {
                    return false;
                }
                next_header.clear();
                tool_results = true;
                return true;
            case conv_item::ASSISTANT:
                break;
        }
        return false;
    }

    // the items of one response go in as one assistant turn, rendered by the chat template
    bool put_response(std::vector<conv_item> & group) {
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
        n_turns++;
        return true;
    }

    void on_append(const json & ev) {
        std::string pcm;
        try {
            pcm = base64::decode(json_value(ev, "audio", std::string()));
        } catch (const std::exception &) {
            send_error("invalid base64 audio");
            return;
        }
        const std::vector<float> in = decode_audio(pcm, in_fmt);
        if (in.empty()) {
            return;
        }
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
        if (!begin_item() || !ensure_turn()) {
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
                        if (commit() && turn_params->create_response) {
                            on_response_create(json::object());
                        }
                    } break;
                case server_turn_action::IDLE_TIMEOUT:
                    {
                        // the user stayed silent: the model takes the turn
                        const std::string item = "item_" + random_string();
                        send_event({{"type", "input_audio_buffer.timeout_triggered"}, {"audio_start_ms", a.audio_ms},
                                    {"audio_end_ms", a.audio_ms}, {"item_id", item}});
                        if (turn_params->create_response && ensure_turn()) {
                            on_response_create(json::object());
                        }
                    } break;
            }
        }
    }

    // ends the audio of the open turn, it becomes a user item
    bool commit() {
        if (!has_audio) {
            send_error("input_audio_buffer.commit: the buffer is empty", "input_audio_buffer_commit_empty");
            return false;
        }
        if (!append_audio(in_rs.flush())) {
            return false;
        }
        // cascade: the turn has its text when the ASR model is done, the client events after the commit wait for it
        std::string asr_text;
        int32_t     asr_in = 0, asr_out = 0;
        bool        asr_ok = true;
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
        has_audio  = false;
        audio_open = false;

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
        const int i = find_item(id);
        if (i >= 0) {
            items[i].item["content"][0]["transcript"] = text;
        }
        send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", id}, {"content_index", 0}, {"delta", text}});
        // the audio embeddings of the prompt are not counted apart, the text of the ASR prompt is small
        send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", id}, {"content_index", 0},
                    {"transcript", text},
                    {"usage", {{"type", "tokens"}, {"total_tokens", n_in + n_out}, {"input_tokens", n_in}, {"output_tokens", n_out},
                               {"input_token_details", {{"text_tokens", 0}, {"audio_tokens", n_in}}}}}});
    }

    // audio.input.transcription: the user audio is transcribed by the ASR model or on another slot, the conversation
    // does not wait for it
    void transcribe_item(const std::string & id, std::vector<float> pcm) {
        const bool by_asr = deps.asr && deps.asr->sample_rate() == model_in_rate;
        if (type != "realtime" || transcription.is_null() || (!by_asr && !deps.transcribe)) {
            return;
        }
        const std::string lang = json_value(transcription, "language", std::string());
        transcribers.emplace_back([this, id, lang, by_asr, pcm = std::move(pcm)]() {
            std::string text;
            int32_t n_in = 0, n_out = 0;
            const bool ok = by_asr ? deps.asr->transcribe(pcm, lang, text, n_in, n_out) :
                                     deps.transcribe(pcm, model_in_rate, lang, text, n_in, n_out);
            if (closed) {
                return;
            }
            std::lock_guard<std::mutex> lock(mutex);
            send_transcription(id, ok, text, n_in, n_out);
        });
    }

    void on_commit() {
        if (!commit() || type != "transcription") {
            return;
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
                        {"transcript", text}});
            items.clear();
            return;
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
            return;
        }
        control(SERVER_CONTROL_ACTION_CLEAR);
        turn_open = false;
        n_turns   = 0;
        std::lock_guard<std::mutex> lock(mutex);
        items.clear();
    }

    // a client that interrupts the assistant cancels the response and truncates it right after: the cancelled
    // response ends within a few frames (its item is in the conversation then), the edit waits for it
    void wait_cancelled_response() {
        for (int i = 0; i < 300 && responding && cancel_sent; i++) {
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
        audio_open = false;
        std::vector<float> pcm = std::move(pending_pcm);
        pending_pcm.clear();
        auto asr = std::move(asr_turn); // cascade: the transcription of the buffer goes on as it is
        asr_turn.reset();
        const bool ok = edit();
        if (!begin_item() || !ensure_turn()) {
            return false;
        }
        audio_open  = true;
        pending_pcm = std::move(pcm);
        asr_turn    = std::move(asr);
        if (!asr_turn && !pending_pcm.empty() && !control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, pending_pcm)) {
            return false;
        }
        return ok;
    }

    int find_item(const std::string & id) {
        for (size_t i = 0; i < items.size(); i++) {
            if (items[i].id == id) {
                return (int) i;
            }
        }
        return -1;
    }

    // goes back to item k and puts items k.. in again, after `change` edited them
    bool rebuild(size_t k, const std::function<void(std::vector<conv_item> &)> & change) {
        std::vector<conv_item> tail;
        std::string            mark;
        turn_state             before = save_turn();
        {
            std::lock_guard<std::mutex> lock(mutex);
            // a response goes back as a whole, from its first item
            while (k > 0 && k < items.size() && items[k].mark.empty()) {
                k--;
            }
            if (k < items.size()) {
                mark   = items[k].mark;
                before = items[k].before;
                tail.assign(std::make_move_iterator(items.begin() + k), std::make_move_iterator(items.end()));
                items.erase(items.begin() + k, items.end());
            }
            last_item_id = items.empty() ? "" : items.back().id;
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

    // after a response with its own context: the session instructions and tools, the conversation as it is
    void restore_conversation() {
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
        std::vector<conv_item> all;
        {
            std::lock_guard<std::mutex> lock(mutex);
            all = std::move(items);
            items.clear();
            last_item_id.clear();
        }
        put_items(all, true);
    }

    void on_item_retrieve(const json & ev) {
        const std::string id = json_value(ev, "item_id", std::string());
        std::lock_guard<std::mutex> lock(mutex);
        const int i = find_item(id);
        if (i < 0) {
            send_error("conversation.item.retrieve: no item " + id);
            return;
        }
        // audio content comes with its audio, in the format the client sent or got it
        json item = items[i].item;
        if (!items[i].audio.empty() && item.contains("content")) {
            for (auto & part : item["content"]) {
                const std::string t = json_value(part, "type", std::string());
                if (t == "input_audio" || t == "output_audio") {
                    part["audio"]  = base64::encode(items[i].audio);
                    part["format"] = items[i].audio_fmt.to_json();
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
        }
        if (i < 0) {
            send_error("conversation.item.delete: no item " + id);
            return;
        }
        if (!can_edit("conversation.item.delete", true)) {
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
        int64_t n_samples = 0;
        int rate = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            i = find_item(id);
            if (i >= 0) {
                n_samples = items[i].audio_samples;
                rate      = items[i].audio_rate;
            }
        }
        if (i < 0) {
            send_error("conversation.item.truncate: no item " + id);
            return;
        }
        if (rate == 0 || part != 0) {
            send_error("conversation.item.truncate: only the audio of an assistant message can be truncated");
            return;
        }
        if (end_ms < 0 || end_ms * rate / 1000 > n_samples) {
            send_error("conversation.item.truncate: audio_end_ms is beyond the audio of the item");
            return;
        }
        if (!can_edit("conversation.item.truncate", true)) {
            return;
        }
        const bool ok = edit_around_audio([&]() { return rebuild(i, [&](std::vector<conv_item> & tail) {
            for (auto & it : tail) {
                if (it.id != id) {
                    continue;
                }
                // the transcript is cut in proportion to the audio heard, at a word boundary
                json & content = it.item["content"];
                std::string tr = content.size() > 0 ? json_value(content[0], "transcript", std::string()) : std::string();
                const int64_t end_s = end_ms * rate / 1000;
                size_t keep = n_samples > 0 ? (size_t) ((double) tr.size() * end_s / n_samples) : tr.size();
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
                it.audio_samples = end_ms * rate / 1000;
                const size_t bps = it.audio_fmt.type == "audio/pcm" ? 2 : 1;
                it.audio.resize(std::min(it.audio.size(), (size_t) it.audio_samples * bps));
            }
        }); });
        if (ok) {
            send_event({{"type", "conversation.item.truncated"}, {"item_id", id}, {"content_index", part}, {"audio_end_ms", end_ms}});
        }
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
        audio_open = false;
        has_audio  = false;
        pending_pcm.clear();
        if (asr_turn) {
            deps.asr->cancel(asr_turn);
            asr_turn.reset();
        }
        speech_item_id.clear();
        in_rs.reset(in_fmt.rate, model_in_rate);
        if (detector && turn_params) {
            detector = std::make_unique<server_turn_detector>(*turn_params, make_vad(), turn_model_for_session());
        }
        send_event({{"type", "input_audio_buffer.cleared"}});
    }

    // an item from the client, as it goes into the conversation
    bool parse_client_item(const json & item, conv_item & it, std::string & err) {
        const std::string type = json_value(item, "type", std::string("message"));
        const std::string role = json_value(item, "role", std::string());
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
        if (type != "message" || (role != "user" && role != "assistant")) {
            err = "only user and assistant messages, function calls and function call outputs are supported";
            return false;
        }
        json content = json::array();
        std::string text;
        for (const auto & p : item.value("content", json::array())) {
            const std::string t = json_value(p, "type", std::string());
            if (role == "user" && t == "input_text") {
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
        if (!it.pcm.empty() && !text.empty()) {
            err = "a user message has either text or audio";
            return false;
        }
        it.kind = role == "assistant" ? conv_item::ASSISTANT : it.pcm.empty() ? conv_item::USER_TEXT : conv_item::USER_AUDIO;
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
        if (!parse_client_item(ev.value("item", json::object()), it, err)) {
            send_error("conversation.item.create: " + err);
            return;
        }
        if (audio_open) {
            send_error("conversation.item.create: the input audio buffer has audio, commit or clear it first");
            return;
        }
        // previous_item_id: insert after that item, the items after it go in again
        if (ev.contains("previous_item_id") && ev["previous_item_id"].is_string() && ev["previous_item_id"] != last_item_id) {
            const std::string prev = ev["previous_item_id"];
            int i;
            {
                std::lock_guard<std::mutex> lock(mutex);
                i = find_item(prev);
            }
            if (i < 0) {
                send_error("conversation.item.create: no item " + prev);
                return;
            }
            if (!can_edit("conversation.item.create")) {
                return;
            }
            const json out = it.item;
            if (rebuild(i + 1, [&](std::vector<conv_item> & tail) { tail.insert(tail.begin(), std::move(it)); })) {
                send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", out}});
                send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", out}});
            }
            return;
        }
        if (it.kind == conv_item::TOOL_OUTPUT) {
            on_tool_output(std::move(it));
            return;
        }
        if (it.kind == conv_item::ASSISTANT) {
            std::vector<conv_item> group;
            group.push_back(std::move(it));
            if (put_response(group)) {
                add_item(std::move(group[0]));
            }
            return;
        }
        if (put_item(it)) {
            const std::string id = it.id;
            std::vector<float> pcm = it.kind == conv_item::USER_AUDIO ? it.pcm : std::vector<float>();
            const std::string text = it.transcript;
            add_item(std::move(it));
            if (pcm.empty()) {
                return;
            }
            if (!cascade()) {
                transcribe_item(id, std::move(pcm));
            } else if (type == "realtime" && !transcription.is_null()) {
                std::lock_guard<std::mutex> lock(mutex);
                send_transcription(id, true, text, 0, 0); // put_item transcribed it
            }
        }
    }

    // the result of a tool call of the last response, the next response answers it
    void on_tool_output(conv_item it) {
        if (!has_tool_text || n_turns == 0 || turn_open) {
            send_error("conversation.item.create: a function call output must follow the response that called it");
            return;
        }
        if (put_item(it)) {
            add_item(std::move(it));
        }
    }

    // the context of a response with its own input, instructions or tools: the conversation (or the input items) from
    // the start, rendered with them; restore_conversation() puts the conversation back once the response is done
    bool put_response_context(const json & resp, bool has_input, const std::string & instr,
                              const std::vector<common_chat_tool> & new_tools, const json & new_tools_json) {
        if (!can_edit("response.create")) {
            return false;
        }
        std::vector<conv_item> ctx_items;
        turn_state             start = save_turn();
        std::string            first_mark;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!items.empty()) {
                start      = items[0].before;
                first_mark = items[0].mark;
            }
            if (!has_input) {
                ctx_items = items;
            }
        }
        if (has_input) {
            for (const auto & in : resp["input"]) {
                if (json_value(in, "type", std::string()) == "item_reference") {
                    std::lock_guard<std::mutex> lock(mutex);
                    const int i = find_item(json_value(in, "id", std::string()));
                    if (i < 0) {
                        send_error("response.create: no item " + json_value(in, "id", std::string()));
                        return false;
                    }
                    ctx_items.push_back(items[i]);
                    ctx_items.back().response_id += "_ref"; // a referenced response item goes in on its own
                    continue;
                }
                conv_item it;
                std::string err;
                if (!parse_client_item(in, it, err)) {
                    send_error("response.create: input: " + err);
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
        if (!first_mark.empty() && !control(SERVER_CONTROL_ACTION_ROLLBACK, first_mark)) {
            send_error("failed to go back in the conversation", "server_error");
            return false;
        }
        restore_mark   = "base" + std::to_string(n_marks++);
        restore_before = start;
        if (!control(SERVER_CONTROL_ACTION_MARK, restore_mark)) {
            return false;
        }
        pending_restore = true;
        load_turn(start);
        if (!put_items(ctx_items, false)) {
            return false;
        }
        // a response needs a user turn or tool results before it
        return turn_open || tool_results || ensure_turn();
    }

    void on_response_create(const json & ev) {
        if (type == "transcription") {
            send_error("response.create: not supported in a transcription session");
            return;
        }
        const json resp = ev.value("response", json::object());
        const std::string conversation = json_value(resp, "conversation", std::string("auto"));
        if (conversation != "auto" && conversation != "none") {
            send_error("response.create: conversation must be \"auto\" or \"none\"");
            return;
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
                send_error("response.create: " + err);
                return;
            }
            resp_tools_json = resp["tools"];
        }
        const bool has_input = resp.contains("input") && resp["input"].is_array();
        const bool fresh     = has_input || resp_instructions != instructions || resp_tools_json != tools_json;

        if (!fresh && n_turns == 0 && !turn_open && !tool_results) {
            send_error("response.create: commit audio or create a message first", "conversation_empty");
            return;
        }
        mtmd_helper_gen_audio_out_mode mode = out_mode;
        if (resp.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(resp["output_modalities"], mode, err)) {
                send_error("response.create: " + err);
                return;
            }
        }
        // with a voice model the session speaks by default, as the OpenAI API
        if (deps.voices && mode == MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            mode = MTMD_HELPER_GEN_AUDIO_OUT_AUDIO;
        }
        if (!can_speak() && mode != MTMD_HELPER_GEN_AUDIO_OUT_TEXT && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate audio, start the server with --voice-model");
            return;
        }
        if (!deps.cap.text_gen && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate text");
            return;
        }
        // audio not committed yet belongs to this turn
        if (has_audio && !commit()) {
            return;
        }
        if (fresh && !put_response_context(resp, has_input, resp_instructions, resp_tools, resp_tools_json)) {
            return;
        }

        common_chat_tool_choice choice = tool_choice;
        if (resp.contains("tool_choice")) {
            std::string err;
            if (!parse_tool_choice(resp["tool_choice"], choice, err)) {
                send_error("response.create: " + err);
                return;
            }
        }

        // the gen-audio pipeline opens the response in the asked mode, after tool results the assistant turn opens right away
        // after a response without new input, the assistant turn follows the previous one
        const std::string prompt = (turn_open ? user_footer : tool_results ? "" : next_header) + assistant_header;
        server_control_response response;
        response.n_predict = max_output_tokens;
        if (resp.contains("max_output_tokens") && !parse_max_output_tokens(resp["max_output_tokens"], response.n_predict)) {
            send_error("response.create: max_output_tokens must be an integer or \"inf\"");
            return;
        }
        // an audio response carries its transcript: the model speaks interleaved with the text it says, or a voice
        // model speaks the text of the model
        const bool use_voice       = deps.voices && (mode == MTMD_HELPER_GEN_AUDIO_OUT_AUDIO || mode == MTMD_HELPER_GEN_AUDIO_OUT_MIXED);
        const bool with_transcript = mode == MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && deps.cap.text_gen;
        response.out_mode = use_voice ? MTMD_HELPER_GEN_AUDIO_OUT_TEXT : with_transcript ? MTMD_HELPER_GEN_AUDIO_OUT_MIXED : mode;
        int resp_voice_model = 0;
        std::string resp_voice_name;
        if (use_voice) {
            resp_voice_model = std::max(0, deps.voices->find(voice == "default" ? "" : voice, resp_voice_name));
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
                    return;
                }
                chat.reset(); // without tools the raw text goes out
            }
        }
        response.sampling = response_sampling(chat ? &*chat : nullptr);

        {
            std::lock_guard<std::mutex> lock(mutex);
            response_id      = "resp_" + random_string();
            response_item_id = "item_" + random_string();
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
            out_rs.reset(use_voice ? deps.voices->models()[resp_voice_model].sample_rate : model_out_rate, out_fmt.rate);
            resp_oob      = oob;
            response_meta = {
                {"conversation_id",   oob ? json(nullptr) : json(conv_id)},
                {"output_modalities", output_modalities_json(mode)},
                {"max_output_tokens", max_output_tokens_json(response.n_predict)},
                {"metadata",          resp.value("metadata", json(nullptr))},
            };
            if (can_speak()) {
                response_meta["audio"] = {{"output", {{"format", out_fmt.to_json()}, {"voice", voice}}}};
            }
            send_event({{"type", "response.created"}, {"response", response_json("in_progress", nullptr, nullptr)}});
        }
        if (!begin_item()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            resp_mark   = pending_mark;
            resp_before = pending_before;
        }
        responding     = true;
        cancel_sent    = false;
        cancel_by_turn = false;
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, prompt, {}, response)) {
            responding = false;
            return;
        }
        if (oob && !fresh) {
            // out of the conversation: back to the mark once the response is done, the slot runs the controls in order
            control(SERVER_CONTROL_ACTION_ROLLBACK, resp_mark);
            load_turn(resp_before);
            return;
        }
        // the response ends the turn, its tokens are in the slot; the footer goes in before the next turn
        next_header  = assistant_footer;
        turn_open    = false;
        tool_results = false;
        n_turns++;
    }

    // the whole update is checked first: a rejected update changes nothing and gets only an error
    void on_session_update(const json & ev) {
        const json session = ev.value("session", json::object());
        auto reject = [this](const std::string & msg) { send_error("session.update: " + msg); };
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

        // new instructions or tools change the text around every turn: the conversation is rebuilt with them
        const bool rerender = started && (new_instructions != instructions || new_tools_json != tools_json);
        if (rerender && !can_edit("session.update")) {
            return;
        }

        // checked: apply
        const auto old_type = type, old_instructions = instructions, old_language = language;
        const auto old_tools = tools;
        type         = new_type;
        instructions = new_instructions;
        language     = new_language;
        tools        = new_tools;
        if ((!started || rerender) && !configure()) {
            type         = old_type;
            instructions = old_instructions;
            language     = old_language;
            tools        = old_tools;
            send_error("the chat template does not fit the realtime turn format", "server_error");
            return;
        }
        tools_json        = new_tools_json;
        out_mode          = new_out_mode;
        tool_choice       = new_choice;
        max_output_tokens = new_max;
        transcription     = new_transcription;
        if (in.contains("noise_reduction")) {
            noise_reduction = in["noise_reduction"];
        }
        if (out.contains("voice") && out["voice"].is_string()) {
            voice = out["voice"];
        }
        if (out.contains("speed") && out["speed"].is_number()) {
            speed = out["speed"].get<double>();
        }
        truncation = session.value("truncation", truncation);
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
            detector.reset();
            if (turn_params) {
                detector = std::make_unique<server_turn_detector>(*turn_params, make_vad(), turn_model_for_session());
            }
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
        }
        const bool ok = derive_turn_text();
        instructions = saved;
        return ok;
    }

    void handle(const std::string & msg) {
        json ev;
        try {
            ev = json::parse(msg);
        } catch (const std::exception &) {
            send_error("invalid JSON");
            return;
        }
        const std::string t = json_value(ev, "type", std::string());
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
                restore_conversation();
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
            cancel_voice();
            cancel_by_turn = false;
            cancel_sent    = true;
            control(SERVER_CONTROL_ACTION_RESPONSE_CANCEL);
        } else {
            send_error("unknown event type: " + t);
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
        const std::string bytes = encode_audio(samples, out_fmt);
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
            if (!res.text.empty()) {
                transcript += res.text;
                send_event({{"type", "conversation.item.input_audio_transcription.delta"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"delta", res.text}});
            }
            if (res.response_done) {
                send_event({{"type", "conversation.item.input_audio_transcription.completed"}, {"item_id", transcribe_item_id},
                            {"content_index", 0}, {"transcript", transcript}});
                transcribe_item_id.clear();
            }
            return;
        }

        // the text goes through the parser: content and tool calls, the reasoning of the model stays out
        std::vector<common_chat_msg_diff> diffs;
        if (parse) {
            if (!res.text.empty() || res.response_done) {
                parse->update_chat_msg(common_chat_input(res.text), !res.response_done, diffs, true);
            }
        } else if (!res.text.empty()) {
            diffs.emplace_back();
            diffs.back().content_delta = res.text;
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
        if (res.cancelled) {
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
            output[i]["status"] = status == "cancelled" ? "incomplete" : status;
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
                it.audio_rate    = out_fmt.rate;
                it.audio         = out_audio;
                it.audio_fmt     = out_fmt;
            }
            items.push_back(std::move(it));
        }
        send_event({{"type", "response.done"}, {"response", response_json(status, details, usage_json(res))}});
        responding     = false;
        response_ended = true;
        voice_resp     = false;
        // the context window of the conversation stands in for the token budget
        send_event({{"type", "rate_limits.updated"}, {"rate_limits", json::array({{
            {"name", "tokens"}, {"limit", res.n_ctx}, {"remaining", std::max(0, res.n_ctx - res.n_past)}, {"reset_seconds", 0},
        }})}});
        parse.reset();
    }

    //
    // session
    //

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
            task.params.sampling          = deps.sampling;
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
                    send_error(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event({{"type", "session.created"}, {"session", session_json()}});

        // client events are handled on this thread, results are sent from the loop below
        std::thread reader([this]() {
            std::string msg;
            while (conn.read(msg) != 0) {
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
                send_error(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * res = dynamic_cast<server_task_result_realtime *>(result.get())) {
                try {
                    on_result(*res);
                } catch (const std::exception & e) {
                    send_error(e.what(), "server_error");
                    break;
                }
            }
        }

        closed = true;
        conn.close();
        reader.join();
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
                                           const common_params_sampling & sampling, server_response & queue_results, int id_task, int index)
    : ctx(ctx), model(llama_get_model(ctx)), vocab(llama_model_get_vocab(llama_get_model(ctx))), mctx(mctx), seq_id(seq_id),
      n_ctx(n_ctx), sampling(sampling), queue_results(queue_results), id_task(id_task), index(index) {
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

void server_realtime_slot::send_result(std::string text, std::vector<float> audio, bool response_done) {
    auto res = std::make_unique<server_task_result_realtime>();
    res->id            = id_task;
    res->index         = index;
    res->text          = std::move(text);
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
bool server_realtime_slot::decode(const llama_token * tokens, const float * rows, int32_t n, bool logits_last) {
    const int32_t n_embd  = llama_model_n_embd_inp(model);
    const int32_t n_batch = llama_n_batch(ctx);
    if (n_past + n >= n_ctx) {
        return fail("the conversation does not fit in the context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
    }
    for (int32_t i0 = 0; i0 < n; i0 += n_batch) {
        const int32_t n_cur = std::min(n_batch, n - i0);
        llama_batch b = llama_batch_init(n_cur, rows ? n_embd : 0, 1);
        for (int32_t i = 0; i < n_cur; i++) {
            if (rows) {
                memcpy(b.embd + (size_t) i * n_embd, rows + (size_t) (i0 + i) * n_embd, n_embd * sizeof(float));
            } else {
                b.token[i] = tokens[i0 + i];
            }
            b.pos[i]       = n_past + i0 + i;
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
    n_past += n;
    if (rows) {
        n_audio += n;
    }
    return true;
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
    }
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

    if (gen.valid()) {
        // the gen-audio pipeline continues the conversation and follows the model's text/audio switching
        mtmd_helper::gen_audio::inp inp;
        inp.set_prompt(op.text);
        inp.data.seq_id   = seq_id;
        inp.data.n_past   = n_past;
        inp.data.seed     = sampling.seed;
        inp.data.stream   = true;
        inp.data.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        inp.data.out_mode = op.response.out_mode;
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
        const llama_tokens toks = common_tokenize(vocab, op.text, false, true);
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
    bool        stop = false;
    std::string text;
    std::vector<float> audio;

    if (gen.valid()) {
        const float * h_next = nullptr;
        if (gen.step_gen(sampled, h_state.data(), &h_next, &stop) != 0) {
            return fail("response generation failed");
        }
        if (sampled != LLAMA_TOKEN_NULL && !stop) {
            text = piece(sampled);
        }
        (sampled != LLAMA_TOKEN_NULL ? n_out_text : n_out_audio)++;
        if (h_next) {
            h_state.assign(h_next, h_next + llama_model_n_embd(model));
        }
    } else {
        stop = llama_vocab_is_eog(vocab, sampled);
        if (!stop) {
            text = piece(sampled);
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
    if (!text.empty() || !audio.empty()) {
        send_result(std::move(text), std::move(audio));
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
    if (n_past > 0) {
        m.state.resize(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (llama_state_seq_get_data_ext(ctx, m.state.data(), m.state.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != m.state.size()) {
            return fail("failed to save the conversation state");
        }
    }
    marks.push_back(std::move(m));
    return true;
}

bool server_realtime_slot::rollback(const std::string & name) {
    auto it = std::find_if(marks.begin(), marks.end(), [&](const mark & m) { return m.name == name; });
    if (it == marks.end()) {
        return fail("unknown conversation mark " + name);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (it->n_past == 0) {
        llama_memory_seq_rm(mem, seq_id, -1, -1);
    } else {
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
    turn_pcm.clear();
    marks.erase(it, marks.end());
    return true;
}

bool server_realtime_slot::update() {
    // the hidden states drive the audio of a model that speaks itself; a text model needs no embeddings
    llama_set_embeddings(ctx, gen.valid());

    if (responding) {
        return response_step();
    }

    while (!ops.empty() && !responding) {
        server_realtime_op op = std::move(ops.front());
        ops.pop_front();
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
                    mtmd_audio_stream_push(stream.get(), op.audio.data(), op.audio.size());
                    if (!audio_rows()) {
                        return false;
                    }
                } break;
            case SERVER_CONTROL_ACTION_AUDIO_COMMIT:
            case SERVER_CONTROL_ACTION_AUDIO_END:
                {
                    // the end of the audio of a turn
                    if (!streaming && !audio_whole()) {
                        return false;
                    }
                    if (stream) {
                        mtmd_audio_stream_flush(stream.get(), true);
                        if (!audio_rows()) {
                            return false;
                        }
                        stream.reset();
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
                    turn_pcm.clear();
                    marks.clear();
                    n_past     = 0;
                    n_audio    = 0;
                    n_prev_end = 0;
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
            default:
                break;
        }
    }
    return true;
}
