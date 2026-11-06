#include "server-realtime.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"
#include "server-turn.h"

#include "base64.hpp"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
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

    // conversation state, owned by the reader thread
    bool        started   = false; // something went into the slot, the session config is fixed
    bool        turn_open = false; // a user turn is being filled
    bool        has_audio = false; // the open turn has audio not committed yet
    bool        tool_results = false; // tool results went in after the last response, the next one answers them
    int         n_turns   = 0;
    std::string next_header;       // text that goes in before the next user turn

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
    int         idx_text  = -1;
    int         idx_audio = -1;
    json        parts     = json::array();
    json        output    = json::array(); // items of the response
    int         idx_message = -1;          // output index of the message, added with its first delta
    std::map<size_t, int> call_items;      // tool call -> output index
    std::optional<task_result_state> parse; // the tool call parser, with tools

    std::atomic<bool> closed{false};

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
        if (deps.cap.audio_gen) {
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

        // a tool result follows the assistant turn that called the tool
        if (!tools.empty()) {
            const std::string result = "@@RESULT@@";
            common_chat_msg call = msg("assistant", "");
            call.tool_calls.push_back({tools[0].name, "{}", "call_0"});
            common_chat_msg res = msg("tool", result);
            res.tool_call_id = "call_0";
            const std::string e  = with({msg("user", user), call});
            const std::string f  = with({msg("user", user), call, res});
            const size_t      pf = f.find(result, e.size());
            if (pf == std::string::npos || f.compare(0, e.size(), e) != 0) {
                return false;
            }
            tool_header = f.substr(e.size(), pf - e.size());
            tool_footer = f.substr(pf + result.size());
        }
        return true;
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
        if (!ensure_turn()) {
            return;
        }
        has_audio = true; // samples held back by the resampler come with the next audio or the commit
        append_audio(std::move(samples));
    }

    bool append_audio(std::vector<float> samples) {
        if (samples.empty()) {
            return true;
        }
        if (!ensure_turn() || !control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, std::move(samples))) {
            return false;
        }
        has_audio = true;
        return true;
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
        for (auto & a : detector->push(samples.data(), samples.size())) {
            switch (a.kind) {
                case server_turn_action::SPEECH_STARTED:
                    {
                        speech_item_id = "item_" + random_string();
                        if (turn_params->interrupt_response && responding) {
                            control(SERVER_CONTROL_ACTION_RESPONSE_CANCEL);
                        }
                        send_event({{"type", "input_audio_buffer.speech_started"}, {"audio_start_ms", a.audio_ms},
                                    {"item_id", speech_item_id}});
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
        std::vector<float> tail = in_rs.flush();
        if (!tail.empty() && !control(SERVER_CONTROL_ACTION_AUDIO_APPEND, {}, std::move(tail))) {
            return false;
        }
        if (!control(SERVER_CONTROL_ACTION_AUDIO_END)) {
            return false;
        }
        has_audio = false;

        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = speech_item_id.empty() ? "item_" + random_string() : speech_item_id;
        speech_item_id.clear();
        send_event({{"type", "input_audio_buffer.committed"}, {"previous_item_id", prev}, {"item_id", last_item_id}});
        const json item = {{"id", last_item_id}, {"type", "message"}, {"status", "completed"}, {"role", "user"},
                           {"content", json::array({{{"type", "input_audio"}, {"transcript", nullptr}}})}};
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", item}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", item}});
        return true;
    }

    void on_commit() {
        if (!commit() || type != "transcription") {
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
    }

    void on_item_create(const json & ev) {
        if (type == "transcription") {
            send_error("conversation.item.create: not supported in a transcription session");
            return;
        }
        const json item = ev.value("item", json::object());
        if (json_value(item, "type", std::string()) == "function_call_output") {
            on_tool_output(item);
            return;
        }
        if (json_value(item, "type", std::string()) != "message" || json_value(item, "role", std::string()) != "user") {
            send_error("conversation.item.create: only user messages and function call outputs are supported");
            return;
        }
        std::string text;
        for (const auto & part : item.value("content", json::array())) {
            const std::string t = json_value(part, "type", std::string());
            if (t != "input_text") {
                send_error("conversation.item.create: unsupported content type: " + t);
                return;
            }
            text += json_value(part, "text", std::string());
        }
        if (!ensure_turn() || !control(SERVER_CONTROL_ACTION_TEXT_APPEND, text)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = item.contains("id") && item["id"].is_string() ? item["id"].get<std::string>() : "item_" + random_string();
        const json out = {{"id", last_item_id}, {"type", "message"}, {"status", "completed"}, {"role", "user"},
                          {"content", json::array({{{"type", "input_text"}, {"text", text}}})}};
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", out}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", out}});
    }

    // the result of a tool call of the last response, the next response answers it
    void on_tool_output(const json & item) {
        if (tools.empty() || n_turns == 0 || turn_open) {
            send_error("conversation.item.create: a function call output must follow the response that called it");
            return;
        }
        const std::string output = json_value(item, "output", std::string());
        if (!control(SERVER_CONTROL_ACTION_TEXT_APPEND, next_header + tool_header + output + tool_footer)) {
            return;
        }
        next_header.clear();
        tool_results = true;

        std::lock_guard<std::mutex> lock(mutex);
        const json prev = last_item_id.empty() ? json(nullptr) : json(last_item_id);
        last_item_id = item.contains("id") && item["id"].is_string() ? item["id"].get<std::string>() : "item_" + random_string();
        const json out = {{"id", last_item_id}, {"type", "function_call_output"},
                          {"call_id", json_value(item, "call_id", std::string())}, {"output", output}};
        send_event({{"type", "conversation.item.added"}, {"previous_item_id", prev}, {"item", out}});
        send_event({{"type", "conversation.item.done"},  {"previous_item_id", prev}, {"item", out}});
    }

    void on_response_create(const json & ev) {
        if (type == "transcription") {
            send_error("response.create: not supported in a transcription session");
            return;
        }
        if (!turn_open && !tool_results) {
            send_error("response.create: commit audio or create a message first", "conversation_empty");
            return;
        }
        mtmd_helper_gen_audio_out_mode mode = out_mode;
        const json resp = ev.value("response", json::object());
        if (resp.contains("output_modalities")) {
            std::string err;
            if (!parse_output_modalities(resp["output_modalities"], mode, err)) {
                send_error("response.create: " + err);
                return;
            }
        }
        if (!deps.cap.audio_gen && mode != MTMD_HELPER_GEN_AUDIO_OUT_TEXT && mode != MTMD_HELPER_GEN_AUDIO_OUT_AUTO) {
            send_error("response.create: the model cannot generate audio");
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

        common_chat_tool_choice choice = tool_choice;
        if (resp.contains("tool_choice")) {
            std::string err;
            if (!parse_tool_choice(resp["tool_choice"], choice, err)) {
                send_error("response.create: " + err);
                return;
            }
        }

        // the gen-audio pipeline opens the response in the asked mode, after tool results the assistant turn opens right away
        const std::string prompt = (turn_open ? user_footer : "") + assistant_header;
        server_control_response response;
        response.n_predict = max_output_tokens;
        if (resp.contains("max_output_tokens") && !parse_max_output_tokens(resp["max_output_tokens"], response.n_predict)) {
            send_error("response.create: max_output_tokens must be an integer or \"inf\"");
            return;
        }
        // an audio response carries its transcript: the model speaks interleaved with the text it says
        const bool with_transcript = mode == MTMD_HELPER_GEN_AUDIO_OUT_AUDIO && deps.cap.text_gen;
        response.out_mode = with_transcript ? MTMD_HELPER_GEN_AUDIO_OUT_MIXED : mode;

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
            out_rs.reset(model_out_rate, out_fmt.rate);
            response_meta = {
                {"conversation_id",   conv_id},
                {"output_modalities", output_modalities_json(mode)},
                {"max_output_tokens", max_output_tokens_json(response.n_predict)},
                {"metadata",          resp.value("metadata", json(nullptr))},
            };
            if (deps.cap.audio_gen) {
                response_meta["audio"] = {{"output", {{"format", out_fmt.to_json()}, {"voice", voice}}}};
            }
            send_event({{"type", "response.created"}, {"response", response_json("in_progress", nullptr, nullptr)}});
        }
        responding = true;
        if (!control(SERVER_CONTROL_ACTION_RESPONSE_CREATE, prompt, {}, response)) {
            responding = false;
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
            if (started && new_instructions != instructions) {
                reject("instructions cannot change after the first audio or message");
                return;
            }
        }
        mtmd_helper_gen_audio_out_mode new_out_mode = out_mode;
        if (session.contains("output_modalities") && !parse_output_modalities(session["output_modalities"], new_out_mode, err)) {
            reject(err);
            return;
        }
        std::vector<common_chat_tool> new_tools = tools;
        json new_tools_json = tools_json;
        if (session.contains("tools") && session["tools"] != tools_json) {
            if (started) {
                reject("tools cannot change after the first audio or message");
                return;
            }
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

        // checked: apply
        const auto old_type = type, old_instructions = instructions, old_language = language;
        const auto old_tools = tools;
        type         = new_type;
        instructions = new_instructions;
        language     = new_language;
        tools        = new_tools;
        if (!started && !configure()) {
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
                detector = std::make_unique<server_turn_detector>(*turn_params, make_vad(), nullptr);
            }
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
        if (t == "session.update" || t == "transcription_session.update") {
            on_session_update(ev);
        } else if (t == "input_audio_buffer.append") {
            on_append(ev);
        } else if (t == "input_audio_buffer.commit") {
            on_commit();
        } else if (t == "input_audio_buffer.clear") {
            // the audio already went into the conversation
            in_rs.reset(in_fmt.rate, model_in_rate);
            send_event({{"type", "input_audio_buffer.cleared"}});
        } else if (t == "conversation.item.create") {
            on_item_create(ev);
        } else if (t == "response.create") {
            on_response_create(ev);
        } else if (t == "response.cancel") {
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

    void on_text(const std::string & text) {
        if (transcript_mode) {
            if (idx_audio < 0) {
                idx_audio = open_part(true);
            }
            response_transcript += text;
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
        json ev = part_base(idx_audio);
        ev["type"]  = "response.output_audio.delta";
        ev["delta"] = base64::encode(encode_audio(samples, out_fmt));
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
        if (idx_audio >= 0) {
            on_audio(out_rs.flush());
        }

        std::string status       = "completed";
        json        details      = nullptr;
        if (res.cancelled) {
            status  = "cancelled";
            details = {{"type", "cancelled"}, {"reason", "client_cancelled"}};
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
            send_event({{"type", "conversation.item.done"}, {"previous_item_id", i > 0 ? output[i - 1]["id"] : json(nullptr)},
                        {"item", output[i]}});
            send_event({{"type", "response.output_item.done"}, {"response_id", response_id}, {"output_index", i}, {"item", output[i]}});
        }
        send_event({{"type", "response.done"}, {"response", response_json(status, details, usage_json(res))}});
        responding     = false;
        response_ended = true;
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
        model_in_rate = mtmd_get_audio_sample_rate(deps.mctx);
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
    if (mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE) {
        gen.init(ctx, mctx);
    }
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
        if (!decode(nullptr, mtmd_audio_stream_get_output_embd(stream.get()), n, false)) {
            return false;
        }
    }
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

bool server_realtime_slot::update() {
    llama_set_embeddings(ctx, true);

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
                    n_past     = 0;
                    n_audio    = 0;
                    n_prev_end = 0;
                } break;
            default:
                break;
        }
    }
    return true;
}
