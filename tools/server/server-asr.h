#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// speech recognition of the realtime API: a speech-to-text model that runs in the server process, next to the
// language model, so that a model without audio input (any text model) can hold a realtime voice conversation
//
// a model with an audio stream and an "asr" chat template (e.g. LFM3-Audio, Voxtral Realtime) transcribes while the
// user speaks, the text is ready when the turn ends; any other audio model of libmtmd (e.g. Qwen3-ASR, Voxtral,
// Gemma 4) transcribes the audio of a turn when it ends. The model has its own llama context and a worker thread.

struct common_params;

// --asr-model model=FILE,mmproj=FILE[,lang=LANG]
struct server_asr_params {
    std::string model;
    std::string mmproj;
    std::string lang; // default language of the transcription, empty: the model detects it
};

// parses the --asr-model spec, throws std::invalid_argument
server_asr_params server_asr_params_parse(const std::string & spec);

// the transcription of one user turn
struct server_asr_turn;

class server_asr_model;

class server_asr {
  public:
    // loads the model (CPU threads and GPU layers of the base params); throws std::runtime_error
    server_asr(const server_asr_params & params, const common_params & base);
    ~server_asr();

    const std::string & name() const;
    int  sample_rate() const;
    bool streaming() const; // transcribes as the audio arrives

    // the text of the turn as it is recognized (streaming models only); runs on the worker thread
    using delta_fn = std::function<void(const std::string & text)>;
    // the text of the whole turn, always called once (ok = false on error or cancel); runs on the worker thread
    using done_fn = std::function<void(bool ok, const std::string & text, int32_t n_input, int32_t n_output)>;

    // starts a turn; language empty: the default of the model
    std::shared_ptr<server_asr_turn> open(const std::string & language, delta_fn on_delta = nullptr);
    // mono PCM at sample_rate()
    void push(const std::shared_ptr<server_asr_turn> & turn, std::vector<float> pcm);
    // the end of the audio of the turn
    void finish(const std::shared_ptr<server_asr_turn> & turn, done_fn on_done);
    // drops the turn
    void cancel(const std::shared_ptr<server_asr_turn> & turn);

    // transcribes PCM at sample_rate() and waits for the text; false on error
    bool transcribe(const std::vector<float> & pcm, const std::string & language, std::string & text,
                    int32_t & n_input, int32_t & n_output);

  private:
    std::unique_ptr<server_asr_model> impl;
};
