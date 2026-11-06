#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// turn detection of a realtime session (turn_detection of the OpenAI realtime API)
//
// the detector splits the input audio into turns: before speech it only keeps the last prefix_padding_ms, from the
// start of speech the audio belongs to the turn, a long enough silence (server_vad) or the end of the turn (semantic_vad)
// ends it. The speech probability comes from a VAD, the end of a turn optionally from a turn model.

struct server_turn_params {
    std::string type                = "server_vad"; // server_vad | semantic_vad
    float       threshold           = 0.5f;
    int         prefix_padding_ms   = 300;
    int         silence_duration_ms = 500;
    int         idle_timeout_ms     = -1; // -1: none
    bool        create_response     = true;
    bool        interrupt_response  = true;
    std::string eagerness           = "auto"; // semantic_vad: low | medium | high | auto
};

// speech probability of fixed-size frames
struct server_vad {
    virtual ~server_vad() = default;
    virtual int   sample_rate() const = 0;
    virtual int   frame_size() const = 0; // samples
    virtual float speech_prob(const float * frame) = 0;
    virtual void  reset() {}
};

// level-based VAD with an adaptive noise floor, used when no VAD model is loaded
std::unique_ptr<server_vad> server_vad_energy(int sample_rate);

// Silero VAD (16 kHz): the weights are loaded once and shared, each session gets its own state
struct server_vad_model;
std::shared_ptr<server_vad_model> server_vad_model_load(const std::string & path); // null on error
std::unique_ptr<server_vad>       server_vad_silero(std::shared_ptr<server_vad_model> model);

// probability that the speaker finished their turn, from the audio of the turn
struct server_turn_model {
    virtual ~server_turn_model() = default;
    virtual int   sample_rate() const = 0;
    virtual float end_of_turn_prob(const float * pcm, size_t n) = 0;
};

struct server_turn_action {
    enum kind_t { AUDIO, SPEECH_STARTED, SPEECH_STOPPED, IDLE_TIMEOUT } kind;
    int64_t            audio_ms = 0; // start (with the prefix padding) or end of speech, or the time of the timeout
    std::vector<float> audio;        // AUDIO: samples of the turn
};

class server_turn_detector {
  public:
    // vad must run at sample_rate, the turn model (may be null) too
    server_turn_detector(const server_turn_params & params, std::unique_ptr<server_vad> vad, server_turn_model * turn_model);

    // audio of the session at the VAD rate; the actions come in order: audio before the start of speech never comes out
    std::vector<server_turn_action> push(const float * pcm, size_t n);

    // a response ended: the idle timeout counts from here
    void response_done();

    bool in_speech() const { return speaking; }

    const server_turn_params & params() const { return prm; }

  private:
    void frame(const float * f, std::vector<server_turn_action> & out);
    bool turn_ended();

    server_turn_params                 prm;
    std::unique_ptr<server_vad>        vad;
    server_turn_model *                turn_model;
    int                                rate;
    int                                frame_n;
    std::vector<float>                 pending; // samples of an incomplete frame
    std::deque<float>                  prefix;  // the last prefix_padding_ms before speech
    std::vector<float>                 turn;    // the last 8 s of the turn, for the turn model
    int64_t                            n_samples    = 0; // samples seen in the session
    int64_t                            silence_n    = 0; // samples of silence in the turn
    int64_t                            speech_n     = 0; // samples of speech before the start is confirmed
    int64_t                            idle_n       = 0; // samples without speech since the last turn or response
    int64_t                            last_check_n = 0; // silence at the last turn model check
    bool                               speaking     = false;
    std::vector<float>                 candidate; // frames of speech before the start is confirmed
};
