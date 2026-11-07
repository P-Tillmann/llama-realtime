#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// voices of the realtime API: text-to-speech models that run in the server process, next to the language model
//
// a response of a model that writes text (e.g. Gemma 4) is spoken sentence by sentence; any audio generation model of
// libmtmd works (Pocket TTS, Soprano, Qwen3-TTS, Chatterbox, LFM3-Audio, ...). Each voice model has its own llama
// context and a worker thread, the jobs of a model run in the order they come.

struct common_params;

// --voice-model model=FILE,mmproj=FILE[,name=NAME][,lang=LANG][,voice=VOICE][,speaker=FILE]
struct server_voice_params {
    std::string model;
    std::string mmproj;
    std::string name;    // what sessions select it by (default: from the file name)
    std::string lang;    // passed to the model, e.g. for Qwen3-TTS, LFM3-Audio, Chatterbox Multilingual
    std::string voice;   // default voice: a predefined voice of the model, or a voice name of its tts template
    std::string speaker; // reference audio to clone
};

// parses the --voice-model spec, throws std::invalid_argument
server_voice_params server_voice_params_parse(const std::string & spec);

// a piece of text to speak; the callbacks run on the worker thread of the model
struct server_voice_job {
    std::string text;
    std::string voice; // empty: the default voice of the model
    std::shared_ptr<std::atomic<bool>> cancelled;
    std::function<void(const float * pcm, size_t n)> on_audio; // mono PCM at the sample rate of the model
    std::function<void(bool ok, int32_t n_frames)>    on_done;  // always called once, also for a cancelled job
};

class server_voice_model;

class server_voices {
  public:
    // loads all models (CPU threads and GPU layers of the base params); throws std::runtime_error
    server_voices(const std::vector<server_voice_params> & params, const common_params & base);
    ~server_voices();

    struct info {
        std::string              name;
        std::string              lang;
        int                      sample_rate;
        std::vector<std::string> voices; // predefined voices, the default voice first
    };
    const std::vector<info> & models() const { return infos; }

    // the model and voice of "name", "voice" or "name/voice"; empty string: the default; -1 if unknown
    int find(const std::string & selector, std::string & voice) const;

    void submit(int model, server_voice_job job);

  private:
    std::vector<std::unique_ptr<server_voice_model>> voice_models;
    std::vector<info>                                infos;
};

// splits the text of a response into pieces to speak: whole sentences, the first one may end at a comma
struct server_voice_segmenter {
    std::string buf;
    bool        first = true;

    struct piece {
        std::string text;    // to speak, formatting left out
        size_t      raw_len; // characters of the response text it covers
    };
    // the pieces that are complete after this text
    std::vector<piece> push(const std::string & text);
    // the rest, at the end of the response
    piece flush();
};
