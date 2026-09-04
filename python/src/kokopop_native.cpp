#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "kokopop.h"

#include "core/backend_names.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

PyObject * KokopopError = nullptr;
PyTypeObject make_type_object() {
    PyTypeObject type{};
    Py_SET_TYPE(&type, &PyType_Type);
    type.ob_base.ob_base.ob_refcnt = 1;
    type.ob_base.ob_size = 0;
    return type;
}

PyTypeObject ModelType = make_type_object();
PyTypeObject AudioType = make_type_object();
PyTypeObject AudioChunkType = make_type_object();
PyTypeObject SynthesisSessionType = make_type_object();
PyTypeObject AudioEncoderType = make_type_object();
PyTypeObject StreamIteratorType = make_type_object();

struct ModelObject {
    PyObject_HEAD
    kokopop_model * handle;
};

struct AudioObject {
    PyObject_HEAD
    kokopop_audio audio;
    Py_ssize_t shape[1];
    Py_ssize_t strides[1];
    int32_t chunk_index;
    int32_t is_final;
};

struct SynthesisSessionObject {
    PyObject_HEAD
    kokopop_synthesis * handle;
    PyObject * model_owner;
};

struct AudioEncoderObject {
    PyObject_HEAD
    kokopop_audio_encoder * handle;
};

struct StreamIteratorObject {
    PyObject_HEAD
    kokopop_synthesis * handle;
    PyObject * model_owner;
    size_t max_chunks;
    PyObject * pending;
    Py_ssize_t pending_index;
    int done;
};

int parse_backend(const char * backend, int32_t & out) {
    const char * value = backend ? backend : "auto";
    if (std::strcmp(value, "auto") == 0) {
        out = KOKOPOP_BACKEND_AUTO;
        return 0;
    }
    // backend_from_name deliberately rejects "auto": it is the default, not a
    // backend the CLI tools can be pinned to. Python accepts it explicitly.
    if (!kokopop::backend_from_name(value, out)) {
        const std::string message =
            std::string("backend must be one of: auto|") + kokopop::backend_name_list();
        PyErr_SetString(PyExc_ValueError, message.c_str());
        return -1;
    }
    return 0;
}

int parse_mode(const char * mode, int32_t & out) {
    const std::string value = mode ? mode : "adaptative";
    if (value == "adaptative" || value == "adaptive") {
        out = KOKOPOP_SYNTH_ADAPTATIVE;
    } else if (value == "long_form" || value == "long-form") {
        out = KOKOPOP_SYNTH_LONG_FORM;
    } else {
        PyErr_SetString(PyExc_ValueError, "mode must be one of: adaptative, long_form");
        return -1;
    }
    return 0;
}

int parse_audio_format(const char * format, int32_t & out) {
    const std::string value = format ? format : "pcm_f32le";
    if (value == "pcm_f32le" || value == "pcm") {
        out = KOKOPOP_AUDIO_PCM_F32LE;
    } else if (value == "wav" || value == "wav_pcm16") {
        out = KOKOPOP_AUDIO_WAV_PCM16;
    } else if (value == "ogg_opus" || value == "opus") {
        out = KOKOPOP_AUDIO_OGG_OPUS;
    } else {
        PyErr_SetString(PyExc_ValueError, "format must be one of: pcm_f32le, wav, ogg_opus");
        return -1;
    }
    return 0;
}

void raise_kokopop_error(int rc) {
    const char * msg = kokopop_last_error();
    if (msg == nullptr || msg[0] == '\0') {
        msg = "kokopop call failed";
    }
    if (rc == KOKOPOP_ERROR_INVALID_ARGUMENT) {
        PyErr_SetString(PyExc_ValueError, msg);
        return;
    }
    PyObject * exc = PyObject_CallFunction(KokopopError, "s", msg);
    if (exc == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, msg);
        return;
    }
    PyObject * code = PyLong_FromLong(rc);
    if (code != nullptr) {
        PyObject_SetAttrString(exc, "code", code);
        Py_DECREF(code);
    }
    PyErr_SetObject(KokopopError, exc);
    Py_DECREF(exc);
}

PyObject * bytes_from_kokopop(kokopop_bytes & bytes) {
    PyObject * out = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(bytes.data),
        static_cast<Py_ssize_t>(bytes.size));
    kokopop_bytes_free(&bytes);
    return out;
}

void audio_refresh_buffer(AudioObject * self) {
    self->shape[0] = static_cast<Py_ssize_t>(self->audio.n_samples);
    self->strides[0] = static_cast<Py_ssize_t>(sizeof(float));
}

PyObject * audio_from_owned(kokopop_audio audio, PyTypeObject * type) {
    auto * obj = PyObject_New(AudioObject, type);
    if (obj == nullptr) {
        kokopop_audio_free(&audio);
        return nullptr;
    }
    obj->audio = audio;
    obj->chunk_index = 0;
    obj->is_final = 1;
    audio_refresh_buffer(obj);
    return reinterpret_cast<PyObject *>(obj);
}

PyObject * chunk_from_c(const kokopop_audio_chunk & chunk) {
    kokopop_audio audio{};
    audio.n_samples = chunk.n_samples;
    audio.sample_rate = chunk.sample_rate;
    if (chunk.n_samples > 0) {
        audio.samples = static_cast<float *>(std::malloc(chunk.n_samples * sizeof(float)));
        if (audio.samples == nullptr) {
            PyErr_NoMemory();
            return nullptr;
        }
        std::memcpy(audio.samples, chunk.samples, chunk.n_samples * sizeof(float));
    }
    auto * obj = reinterpret_cast<AudioObject *>(audio_from_owned(audio, &AudioChunkType));
    if (obj == nullptr) {
        return nullptr;
    }
    obj->chunk_index = chunk.chunk_index;
    obj->is_final = chunk.is_final;
    return reinterpret_cast<PyObject *>(obj);
}

int get_optional_string(PyObject * kwargs, const char * key, const char * & out) {
    if (kwargs == nullptr) return 0;
    PyObject * value = PyDict_GetItemString(kwargs, key);
    if (value == nullptr) return 0;
    out = PyUnicode_AsUTF8(value);
    if (out == nullptr) return -1;
    return 0;
}

int get_optional_int(PyObject * kwargs, const char * key, int32_t & out) {
    if (kwargs == nullptr) return 0;
    PyObject * value = PyDict_GetItemString(kwargs, key);
    if (value == nullptr || value == Py_None) return 0;
    long v = PyLong_AsLong(value);
    if (PyErr_Occurred()) return -1;
    out = static_cast<int32_t>(v);
    return 0;
}

int get_optional_u32(PyObject * kwargs, const char * key, uint32_t & out) {
    if (kwargs == nullptr) return 0;
    PyObject * value = PyDict_GetItemString(kwargs, key);
    if (value == nullptr || value == Py_None) return 0;
    unsigned long v = PyLong_AsUnsignedLong(value);
    if (PyErr_Occurred()) return -1;
    out = static_cast<uint32_t>(v);
    return 0;
}

int get_optional_u64(PyObject * kwargs, const char * key, uint64_t & out, bool & present) {
    present = false;
    if (kwargs == nullptr) return 0;
    PyObject * value = PyDict_GetItemString(kwargs, key);
    if (value == nullptr || value == Py_None) return 0;
    unsigned long long v = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return -1;
    out = static_cast<uint64_t>(v);
    present = true;
    return 0;
}

int get_optional_float(PyObject * kwargs, const char * key, float & out) {
    if (kwargs == nullptr) return 0;
    PyObject * value = PyDict_GetItemString(kwargs, key);
    if (value == nullptr || value == Py_None) return 0;
    double v = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) return -1;
    out = static_cast<float>(v);
    return 0;
}

int reject_unknown_kwargs(PyObject * kwargs, const char * const * allowed) {
    if (kwargs == nullptr) return 0;
    PyObject * key = nullptr;
    PyObject * value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
        const char * name = PyUnicode_AsUTF8(key);
        if (name == nullptr) return -1;
        bool ok = false;
        for (size_t i = 0; allowed[i] != nullptr; ++i) {
            if (std::strcmp(name, allowed[i]) == 0) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%s'", name);
            return -1;
        }
    }
    return 0;
}

int fill_synthesis_options(PyObject * kwargs, kokopop_synthesis_options & opts) {
    static const char * const allowed[] = {
        "voice", "speed", "mode", "target_min_tokens", "target_max_tokens",
        "soft_max_tokens", "hard_max_tokens", "first_chunk_target_tokens",
        "target_overshoot_tokens", "comma_pause_ms", "sentence_pause_ms",
        "paragraph_pause_ms", "crossfade_ms", "max_silence_trim_ms",
        "trim_silence", "enable_diffusion", "diffusion_seed",
        "diffusion_steps", "diffusion_alpha", "diffusion_beta",
        "diffusion_embedding_scale", "noise_seed", "max_chunks", nullptr,
    };
    if (reject_unknown_kwargs(kwargs, allowed) < 0) return -1;

    const char * mode = "adaptative";
    if (get_optional_string(kwargs, "voice", opts.voice) < 0) return -1;
    if (get_optional_float(kwargs, "speed", opts.speed) < 0) return -1;
    if (get_optional_string(kwargs, "mode", mode) < 0) return -1;
    if (parse_mode(mode, opts.mode) < 0) return -1;

    if (get_optional_int(kwargs, "target_min_tokens", opts.target_min_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "target_max_tokens", opts.target_max_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "soft_max_tokens", opts.soft_max_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "hard_max_tokens", opts.hard_max_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "first_chunk_target_tokens", opts.first_chunk_target_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "target_overshoot_tokens", opts.target_overshoot_tokens) < 0) return -1;
    if (get_optional_int(kwargs, "comma_pause_ms", opts.comma_pause_ms) < 0) return -1;
    if (get_optional_int(kwargs, "sentence_pause_ms", opts.sentence_pause_ms) < 0) return -1;
    if (get_optional_int(kwargs, "paragraph_pause_ms", opts.paragraph_pause_ms) < 0) return -1;
    if (get_optional_int(kwargs, "crossfade_ms", opts.crossfade_ms) < 0) return -1;
    if (get_optional_int(kwargs, "max_silence_trim_ms", opts.max_silence_trim_ms) < 0) return -1;
    if (get_optional_int(kwargs, "enable_diffusion", opts.enable_diffusion) < 0) return -1;
    if (get_optional_u32(kwargs, "diffusion_seed", opts.diffusion_seed) < 0) return -1;
    if (get_optional_int(kwargs, "diffusion_steps", opts.diffusion_steps) < 0) return -1;
    if (get_optional_float(kwargs, "diffusion_alpha", opts.diffusion_alpha) < 0) return -1;
    if (get_optional_float(kwargs, "diffusion_beta", opts.diffusion_beta) < 0) return -1;
    if (get_optional_float(kwargs, "diffusion_embedding_scale", opts.diffusion_embedding_scale) < 0) return -1;

    bool has_noise_seed = false;
    uint64_t noise_seed = 0;
    if (get_optional_u64(kwargs, "noise_seed", noise_seed, has_noise_seed) < 0) return -1;
    opts.has_sano_noise_seed = has_noise_seed ? 1 : 0;
    opts.sano_noise_seed = noise_seed;

    PyObject * trim = kwargs ? PyDict_GetItemString(kwargs, "trim_silence") : nullptr;
    if (trim != nullptr && trim != Py_None) {
        int truth = PyObject_IsTrue(trim);
        if (truth < 0) return -1;
        opts.trim_silence = truth ? 1 : -1;
    }
    return 0;
}

PyObject * chunks_to_list(kokopop_audio_chunk * chunks, size_t n_chunks) {
    PyObject * list = PyList_New(static_cast<Py_ssize_t>(n_chunks));
    if (list == nullptr) {
        kokopop_audio_chunks_free(chunks, n_chunks);
        return nullptr;
    }
    for (size_t i = 0; i < n_chunks; ++i) {
        PyObject * chunk = chunk_from_c(chunks[i]);
        if (chunk == nullptr) {
            Py_DECREF(list);
            kokopop_audio_chunks_free(chunks, n_chunks);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), chunk);
    }
    kokopop_audio_chunks_free(chunks, n_chunks);
    return list;
}

int require_synthesis_open(SynthesisSessionObject * self) {
    if (self->handle != nullptr) return 0;
    PyErr_SetString(PyExc_ValueError, "SynthesisSession is closed");
    return -1;
}

int require_encoder_open(AudioEncoderObject * self) {
    if (self->handle != nullptr) return 0;
    PyErr_SetString(PyExc_ValueError, "AudioEncoder is closed");
    return -1;
}

void stream_iterator_close(StreamIteratorObject * self) {
    kokopop_synthesis_free(self->handle);
    self->handle = nullptr;
    Py_XDECREF(self->model_owner);
    self->model_owner = nullptr;
    Py_XDECREF(self->pending);
    self->pending = nullptr;
    self->pending_index = 0;
    self->done = 1;
}

void Model_dealloc(ModelObject * self) {
    kokopop_model_free(self->handle);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

int Model_init(ModelObject * self, PyObject * args, PyObject * kwargs) {
    const char * path = nullptr;
    int n_threads = 0;
    const char * backend = "auto";
    static const char * kwlist[] = {"path", "n_threads", "backend", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|is", const_cast<char **>(kwlist),
                                     &path, &n_threads, &backend)) {
        return -1;
    }

    int32_t backend_code = KOKOPOP_BACKEND_AUTO;
    if (parse_backend(backend, backend_code) < 0) return -1;

    kokopop_model_options opts{};
    opts.n_threads = static_cast<int32_t>(n_threads);
    opts.backend = backend_code;

    kokopop_model * model = nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_model_load(path, &opts, &model);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return -1;
    }
    self->handle = model;
    return 0;
}

PyObject * Model_get_sample_rate(ModelObject * self, void *) {
    return PyLong_FromLong(kokopop_model_sample_rate(self->handle));
}

PyObject * Model_get_arch(ModelObject * self, void *) {
    return PyUnicode_FromString(kokopop_model_arch_name(self->handle));
}

PyObject * Model_get_voices(ModelObject * self, void *) {
    const size_t n = kokopop_model_voice_count(self->handle);
    PyObject * tuple = PyTuple_New(static_cast<Py_ssize_t>(n));
    if (tuple == nullptr) return nullptr;
    for (size_t i = 0; i < n; ++i) {
        const char * name = kokopop_model_voice_name(self->handle, i);
        PyObject * item = PyUnicode_FromString(name ? name : "");
        if (item == nullptr) {
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), item);
    }
    return tuple;
}

PyObject * Model_voice_sample_rate(ModelObject * self, PyObject * args) {
    const char * voice = nullptr;
    if (!PyArg_ParseTuple(args, "s", &voice)) return nullptr;
    const int32_t rate = kokopop_model_voice_sample_rate(self->handle, voice);
    if (rate == 0) {
        PyErr_Format(PyExc_KeyError, "unknown voice '%s'", voice);
        return nullptr;
    }
    return PyLong_FromLong(rate);
}

PyObject * Model_synthesize_common(ModelObject * self, PyObject * args, PyObject * kwargs, bool phonemes) {
    const char * input = nullptr;
    const char * voice = "";
    float speed = 1.0f;
    static const char * text_kwlist[] = {"text", "voice", "speed", nullptr};
    static const char * phoneme_kwlist[] = {"phonemes", "voice", "speed", nullptr};
    const char ** chosen = phonemes ? phoneme_kwlist : text_kwlist;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|sf", const_cast<char **>(chosen),
                                     &input, &voice, &speed)) {
        return nullptr;
    }

    kokopop_audio audio{};
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    if (phonemes) {
        rc = kokopop_synthesize_phonemes(self->handle, input, voice, speed, &audio);
    } else {
        rc = kokopop_synthesize_text(self->handle, input, voice, speed, &audio);
    }
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    return audio_from_owned(audio, &AudioType);
}

PyObject * Model_synthesize(ModelObject * self, PyObject * args, PyObject * kwargs) {
    return Model_synthesize_common(self, args, kwargs, false);
}

PyObject * Model_synthesize_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return Model_synthesize(reinterpret_cast<ModelObject *>(self), args, kwargs);
}

PyObject * Model_synthesize_phonemes(ModelObject * self, PyObject * args, PyObject * kwargs) {
    return Model_synthesize_common(self, args, kwargs, true);
}

PyObject * Model_synthesize_phonemes_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return Model_synthesize_phonemes(reinterpret_cast<ModelObject *>(self), args, kwargs);
}

PyObject * Model_stream(ModelObject * self, PyObject * args, PyObject * kwargs) {
    const char * text = nullptr;
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "stream() requires exactly one text argument");
        return nullptr;
    }
    text = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 0));
    if (text == nullptr) return nullptr;

    kokopop_synthesis_options opts{};
    opts.voice = "";
    opts.speed = 1.0f;
    opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;
    if (fill_synthesis_options(kwargs, opts) < 0) return nullptr;

    int32_t max_chunks_i = 1;
    if (get_optional_int(kwargs, "max_chunks", max_chunks_i) < 0) return nullptr;
    size_t max_chunks = static_cast<size_t>(std::max<int32_t>(1, max_chunks_i));

    kokopop_synthesis * synthesis = nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_create(self->handle, &opts, &synthesis);
    if (rc == KOKOPOP_OK) rc = kokopop_synthesis_push_text(synthesis, text);
    if (rc == KOKOPOP_OK) rc = kokopop_synthesis_finish_input(synthesis);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        kokopop_synthesis_free(synthesis);
        raise_kokopop_error(rc);
        return nullptr;
    }

    auto * iter = PyObject_New(StreamIteratorObject, &StreamIteratorType);
    if (iter == nullptr) {
        kokopop_synthesis_free(synthesis);
        return nullptr;
    }
    iter->handle = synthesis;
    iter->model_owner = reinterpret_cast<PyObject *>(self);
    Py_INCREF(iter->model_owner);
    iter->max_chunks = max_chunks;
    iter->pending = nullptr;
    iter->pending_index = 0;
    iter->done = 0;
    return reinterpret_cast<PyObject *>(iter);
}

PyObject * Model_stream_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return Model_stream(reinterpret_cast<ModelObject *>(self), args, kwargs);
}

void StreamIterator_dealloc(StreamIteratorObject * self) {
    stream_iterator_close(self);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject * StreamIterator_iternext(StreamIteratorObject * self) {
    if (self->pending != nullptr) {
        Py_ssize_t n_pending = PyList_GET_SIZE(self->pending);
        if (self->pending_index < n_pending) {
            PyObject * item = PyList_GET_ITEM(self->pending, self->pending_index++);
            Py_INCREF(item);
            if (self->pending_index >= n_pending) {
                Py_CLEAR(self->pending);
                self->pending_index = 0;
            }
            return item;
        }
        Py_CLEAR(self->pending);
        self->pending_index = 0;
    }

    if (self->done || self->handle == nullptr) {
        PyErr_SetNone(PyExc_StopIteration);
        return nullptr;
    }

    kokopop_audio_chunk * chunks = nullptr;
    size_t n_chunks = 0;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_next(self->handle, self->max_chunks, &chunks, &n_chunks);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        stream_iterator_close(self);
        raise_kokopop_error(rc);
        return nullptr;
    }
    if (n_chunks == 0) {
        stream_iterator_close(self);
        PyErr_SetNone(PyExc_StopIteration);
        return nullptr;
    }

    PyObject * list = chunks_to_list(chunks, n_chunks);
    if (list == nullptr) {
        stream_iterator_close(self);
        return nullptr;
    }
    auto * last = reinterpret_cast<AudioObject *>(PyList_GET_ITEM(list, static_cast<Py_ssize_t>(n_chunks - 1)));
    if (last->is_final) {
        self->done = 1;
        kokopop_synthesis_free(self->handle);
        self->handle = nullptr;
        Py_CLEAR(self->model_owner);
    }

    PyObject * item = PyList_GET_ITEM(list, 0);
    Py_INCREF(item);
    if (n_chunks > 1) {
        self->pending = list;
        self->pending_index = 1;
    } else {
        Py_DECREF(list);
    }
    return item;
}

PyObject * StreamIterator_close(StreamIteratorObject * self, PyObject *) {
    stream_iterator_close(self);
    Py_RETURN_NONE;
}

void Audio_dealloc(AudioObject * self) {
    kokopop_audio_free(&self->audio);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject * Audio_get_sample_rate(AudioObject * self, void *) {
    return PyLong_FromLong(self->audio.sample_rate);
}

PyObject * Audio_get_n_samples(AudioObject * self, void *) {
    return PyLong_FromSize_t(self->audio.n_samples);
}

PyObject * AudioChunk_get_chunk_index(AudioObject * self, void *) {
    return PyLong_FromLong(self->chunk_index);
}

PyObject * AudioChunk_get_is_final(AudioObject * self, void *) {
    return PyBool_FromLong(self->is_final != 0);
}

int Audio_getbuffer(AudioObject * self, Py_buffer * view, int flags) {
    audio_refresh_buffer(self);
    if (PyBuffer_FillInfo(view, reinterpret_cast<PyObject *>(self), self->audio.samples,
                          static_cast<Py_ssize_t>(self->audio.n_samples * sizeof(float)),
                          1, flags) < 0) {
        return -1;
    }
    view->itemsize = static_cast<Py_ssize_t>(sizeof(float));
    view->format = const_cast<char *>("f");
    view->ndim = 1;
    view->shape = self->shape;
    view->strides = self->strides;
    return 0;
}

PyObject * Audio_to_wav_bytes(AudioObject * self, PyObject *) {
    kokopop_encoder_options opts{};
    opts.format = KOKOPOP_AUDIO_WAV_PCM16;
    opts.sample_rate = self->audio.sample_rate > 0 ? self->audio.sample_rate : 24000;
    kokopop_audio_encoder * encoder = nullptr;
    kokopop_bytes bytes{};
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_audio_encoder_create(&opts, &encoder);
    if (rc == KOKOPOP_OK) rc = kokopop_audio_encoder_push(encoder, self->audio.samples, self->audio.n_samples, 1, &bytes);
    kokopop_bytes_free(&bytes);
    if (rc == KOKOPOP_OK) rc = kokopop_audio_encoder_finish(encoder, 1, &bytes);
    Py_END_ALLOW_THREADS
    kokopop_audio_encoder_free(encoder);
    if (rc != KOKOPOP_OK) {
        kokopop_bytes_free(&bytes);
        raise_kokopop_error(rc);
        return nullptr;
    }
    return bytes_from_kokopop(bytes);
}

PyObject * Audio_write_wav(AudioObject * self, PyObject * args) {
    const char * path = nullptr;
    if (!PyArg_ParseTuple(args, "s", &path)) return nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_write_wav(path, &self->audio);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

void SynthesisSession_dealloc(SynthesisSessionObject * self) {
    kokopop_synthesis_free(self->handle);
    self->handle = nullptr;
    Py_XDECREF(self->model_owner);
    self->model_owner = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

int SynthesisSession_init(SynthesisSessionObject * self, PyObject * args, PyObject * kwargs) {
    if (PyTuple_Size(args) != 1 || !PyObject_TypeCheck(PyTuple_GET_ITEM(args, 0), &ModelType)) {
        PyErr_SetString(PyExc_TypeError, "SynthesisSession() requires a Model as its first argument");
        return -1;
    }
    auto * model = reinterpret_cast<ModelObject *>(PyTuple_GET_ITEM(args, 0));

    kokopop_synthesis_options opts{};
    opts.voice = "";
    opts.speed = 1.0f;
    opts.mode = KOKOPOP_SYNTH_ADAPTATIVE;
    if (fill_synthesis_options(kwargs, opts) < 0) return -1;

    kokopop_synthesis * synthesis = nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_create(model->handle, &opts, &synthesis);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return -1;
    }
    self->handle = synthesis;
    self->model_owner = reinterpret_cast<PyObject *>(model);
    Py_INCREF(self->model_owner);
    return 0;
}

PyObject * SynthesisSession_push_text(SynthesisSessionObject * self, PyObject * args) {
    if (require_synthesis_open(self) < 0) return nullptr;
    const char * text = nullptr;
    if (!PyArg_ParseTuple(args, "s", &text)) return nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_push_text(self->handle, text);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject * SynthesisSession_finish_input(SynthesisSessionObject * self, PyObject *) {
    if (require_synthesis_open(self) < 0) return nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_finish_input(self->handle);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject * SynthesisSession_next(SynthesisSessionObject * self, PyObject * args, PyObject * kwargs) {
    if (require_synthesis_open(self) < 0) return nullptr;
    int max_chunks_i = 1;
    static const char * kwlist[] = {"max_chunks", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", const_cast<char **>(kwlist), &max_chunks_i)) {
        return nullptr;
    }
    size_t max_chunks = static_cast<size_t>(std::max(1, max_chunks_i));
    kokopop_audio_chunk * chunks = nullptr;
    size_t n_chunks = 0;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_synthesis_next(self->handle, max_chunks, &chunks, &n_chunks);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    return chunks_to_list(chunks, n_chunks);
}

PyObject * SynthesisSession_next_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return SynthesisSession_next(reinterpret_cast<SynthesisSessionObject *>(self), args, kwargs);
}

PyObject * SynthesisSession_close(SynthesisSessionObject * self, PyObject *) {
    kokopop_synthesis_free(self->handle);
    self->handle = nullptr;
    Py_CLEAR(self->model_owner);
    Py_RETURN_NONE;
}

PyObject * SynthesisSession_enter(SynthesisSessionObject * self, PyObject *) {
    if (require_synthesis_open(self) < 0) return nullptr;
    Py_INCREF(self);
    return reinterpret_cast<PyObject *>(self);
}

PyObject * SynthesisSession_exit(SynthesisSessionObject * self, PyObject *) {
    return SynthesisSession_close(self, nullptr);
}

void AudioEncoder_dealloc(AudioEncoderObject * self) {
    kokopop_audio_encoder_free(self->handle);
    self->handle = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

int AudioEncoder_init(AudioEncoderObject * self, PyObject * args, PyObject * kwargs) {
    const char * format = "pcm_f32le";
    int sample_rate = 24000;
    int ogg_prebuffer_chunks = 0;
    static const char * kwlist[] = {"format", "sample_rate", "ogg_prebuffer_chunks", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sii", const_cast<char **>(kwlist),
                                     &format, &sample_rate, &ogg_prebuffer_chunks)) {
        return -1;
    }
    kokopop_encoder_options opts{};
    if (parse_audio_format(format, opts.format) < 0) return -1;
    opts.sample_rate = static_cast<int32_t>(sample_rate);
    opts.ogg_prebuffer_chunks = static_cast<int32_t>(ogg_prebuffer_chunks);

    kokopop_audio_encoder * encoder = nullptr;
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_audio_encoder_create(&opts, &encoder);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return -1;
    }
    self->handle = encoder;
    return 0;
}

PyObject * AudioEncoder_start(AudioEncoderObject * self, PyObject *) {
    if (require_encoder_open(self) < 0) return nullptr;
    kokopop_bytes bytes{};
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_audio_encoder_start(self->handle, &bytes);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    return bytes_from_kokopop(bytes);
}

PyObject * AudioEncoder_push(AudioEncoderObject * self, PyObject * args, PyObject * kwargs) {
    if (require_encoder_open(self) < 0) return nullptr;
    PyObject * samples = nullptr;
    int is_final = 0;
    static const char * kwlist[] = {"samples", "is_final", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char **>(kwlist), &samples, &is_final)) {
        return nullptr;
    }

    Py_buffer view{};
    if (PyObject_GetBuffer(samples, &view, PyBUF_CONTIG_RO) < 0) return nullptr;
    if (view.len % static_cast<Py_ssize_t>(sizeof(float)) != 0) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "samples buffer size must be a multiple of float32");
        return nullptr;
    }

    kokopop_bytes bytes{};
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_audio_encoder_push(
        self->handle,
        static_cast<const float *>(view.buf),
        static_cast<size_t>(view.len / static_cast<Py_ssize_t>(sizeof(float))),
        is_final,
        &bytes);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    return bytes_from_kokopop(bytes);
}

PyObject * AudioEncoder_push_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return AudioEncoder_push(reinterpret_cast<AudioEncoderObject *>(self), args, kwargs);
}

PyObject * AudioEncoder_finish(AudioEncoderObject * self, PyObject * args, PyObject * kwargs) {
    if (require_encoder_open(self) < 0) return nullptr;
    int success = 1;
    static const char * kwlist[] = {"success", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", const_cast<char **>(kwlist), &success)) {
        return nullptr;
    }
    kokopop_bytes bytes{};
    int rc = KOKOPOP_OK;
    Py_BEGIN_ALLOW_THREADS
    rc = kokopop_audio_encoder_finish(self->handle, success, &bytes);
    Py_END_ALLOW_THREADS
    if (rc != KOKOPOP_OK) {
        raise_kokopop_error(rc);
        return nullptr;
    }
    return bytes_from_kokopop(bytes);
}

PyObject * AudioEncoder_finish_py(PyObject * self, PyObject * args, PyObject * kwargs) {
    return AudioEncoder_finish(reinterpret_cast<AudioEncoderObject *>(self), args, kwargs);
}

PyObject * AudioEncoder_close(AudioEncoderObject * self, PyObject *) {
    kokopop_audio_encoder_free(self->handle);
    self->handle = nullptr;
    Py_RETURN_NONE;
}

PyObject * AudioEncoder_enter(AudioEncoderObject * self, PyObject *) {
    if (require_encoder_open(self) < 0) return nullptr;
    Py_INCREF(self);
    return reinterpret_cast<PyObject *>(self);
}

PyObject * AudioEncoder_exit(AudioEncoderObject * self, PyObject *) {
    return AudioEncoder_close(self, nullptr);
}

PyMethodDef Model_methods[] = {
    {"synthesize", _PyCFunction_CAST(Model_synthesize_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"synthesize_phonemes", _PyCFunction_CAST(Model_synthesize_phonemes_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"stream", _PyCFunction_CAST(Model_stream_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"voice_sample_rate", reinterpret_cast<PyCFunction>(Model_voice_sample_rate), METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef Model_getset[] = {
    {"sample_rate", reinterpret_cast<getter>(Model_get_sample_rate), nullptr, nullptr, nullptr},
    {"arch", reinterpret_cast<getter>(Model_get_arch), nullptr, nullptr, nullptr},
    {"voices", reinterpret_cast<getter>(Model_get_voices), nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef Audio_methods[] = {
    {"to_wav_bytes", reinterpret_cast<PyCFunction>(Audio_to_wav_bytes), METH_NOARGS, nullptr},
    {"write_wav", reinterpret_cast<PyCFunction>(Audio_write_wav), METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef Audio_getset[] = {
    {"sample_rate", reinterpret_cast<getter>(Audio_get_sample_rate), nullptr, nullptr, nullptr},
    {"n_samples", reinterpret_cast<getter>(Audio_get_n_samples), nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyGetSetDef AudioChunk_getset[] = {
    {"sample_rate", reinterpret_cast<getter>(Audio_get_sample_rate), nullptr, nullptr, nullptr},
    {"n_samples", reinterpret_cast<getter>(Audio_get_n_samples), nullptr, nullptr, nullptr},
    {"chunk_index", reinterpret_cast<getter>(AudioChunk_get_chunk_index), nullptr, nullptr, nullptr},
    {"is_final", reinterpret_cast<getter>(AudioChunk_get_is_final), nullptr, nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyBufferProcs Audio_buffer = {
    reinterpret_cast<getbufferproc>(Audio_getbuffer),
    nullptr,
};

PyMethodDef SynthesisSession_methods[] = {
    {"push_text", reinterpret_cast<PyCFunction>(SynthesisSession_push_text), METH_VARARGS, nullptr},
    {"finish_input", reinterpret_cast<PyCFunction>(SynthesisSession_finish_input), METH_NOARGS, nullptr},
    {"next", _PyCFunction_CAST(SynthesisSession_next_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"close", reinterpret_cast<PyCFunction>(SynthesisSession_close), METH_NOARGS, nullptr},
    {"__enter__", reinterpret_cast<PyCFunction>(SynthesisSession_enter), METH_NOARGS, nullptr},
    {"__exit__", reinterpret_cast<PyCFunction>(SynthesisSession_exit), METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef AudioEncoder_methods[] = {
    {"start", reinterpret_cast<PyCFunction>(AudioEncoder_start), METH_NOARGS, nullptr},
    {"push", _PyCFunction_CAST(AudioEncoder_push_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"finish", _PyCFunction_CAST(AudioEncoder_finish_py), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"close", reinterpret_cast<PyCFunction>(AudioEncoder_close), METH_NOARGS, nullptr},
    {"__enter__", reinterpret_cast<PyCFunction>(AudioEncoder_enter), METH_NOARGS, nullptr},
    {"__exit__", reinterpret_cast<PyCFunction>(AudioEncoder_exit), METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef StreamIterator_methods[] = {
    {"close", reinterpret_cast<PyCFunction>(StreamIterator_close), METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef make_module_def() {
    PyModuleDef module{};
    module.m_base = PyModuleDef_HEAD_INIT;
    module.m_name = "kokopop._native";
    module.m_size = -1;
    return module;
}

PyModuleDef module = make_module_def();

int ready_type(PyTypeObject & type, const char * name, Py_ssize_t basicsize,
               destructor dealloc, PyMethodDef * methods = nullptr,
               PyGetSetDef * getset = nullptr, initproc init = nullptr,
               PyBufferProcs * buffer = nullptr) {
    type.tp_name = name;
    type.tp_basicsize = basicsize;
    type.tp_flags = Py_TPFLAGS_DEFAULT;
    type.tp_new = PyType_GenericNew;
    type.tp_dealloc = dealloc;
    type.tp_methods = methods;
    type.tp_getset = getset;
    type.tp_init = init;
    type.tp_as_buffer = buffer;
    return PyType_Ready(&type);
}

int ready_stream_iterator_type() {
    StreamIteratorType.tp_name = "kokopop._native.StreamIterator";
    StreamIteratorType.tp_basicsize = sizeof(StreamIteratorObject);
    StreamIteratorType.tp_flags = Py_TPFLAGS_DEFAULT;
    StreamIteratorType.tp_dealloc = reinterpret_cast<destructor>(StreamIterator_dealloc);
    StreamIteratorType.tp_methods = StreamIterator_methods;
    StreamIteratorType.tp_iter = PyObject_SelfIter;
    StreamIteratorType.tp_iternext = reinterpret_cast<iternextfunc>(StreamIterator_iternext);
    return PyType_Ready(&StreamIteratorType);
}

} // namespace

PyMODINIT_FUNC PyInit__native(void) {
    if (ready_type(ModelType, "kokopop._native.Model", sizeof(ModelObject),
                   reinterpret_cast<destructor>(Model_dealloc), Model_methods,
                   Model_getset, reinterpret_cast<initproc>(Model_init)) < 0) {
        return nullptr;
    }
    if (ready_type(AudioType, "kokopop._native.Audio", sizeof(AudioObject),
                   reinterpret_cast<destructor>(Audio_dealloc), Audio_methods,
                   Audio_getset, nullptr, &Audio_buffer) < 0) {
        return nullptr;
    }
    if (ready_type(AudioChunkType, "kokopop._native.AudioChunk", sizeof(AudioObject),
                   reinterpret_cast<destructor>(Audio_dealloc), Audio_methods,
                   AudioChunk_getset, nullptr, &Audio_buffer) < 0) {
        return nullptr;
    }
    if (ready_type(SynthesisSessionType, "kokopop._native.SynthesisSession", sizeof(SynthesisSessionObject),
                   reinterpret_cast<destructor>(SynthesisSession_dealloc), SynthesisSession_methods,
                   nullptr, reinterpret_cast<initproc>(SynthesisSession_init)) < 0) {
        return nullptr;
    }
    if (ready_type(AudioEncoderType, "kokopop._native.AudioEncoder", sizeof(AudioEncoderObject),
                   reinterpret_cast<destructor>(AudioEncoder_dealloc), AudioEncoder_methods,
                   nullptr, reinterpret_cast<initproc>(AudioEncoder_init)) < 0) {
        return nullptr;
    }
    if (ready_stream_iterator_type() < 0) {
        return nullptr;
    }

    PyObject * m = PyModule_Create(&module);
    if (m == nullptr) return nullptr;

    KokopopError = PyErr_NewException("kokopop.KokopopError", PyExc_RuntimeError, nullptr);
    if (KokopopError == nullptr) {
        Py_DECREF(m);
        return nullptr;
    }

    Py_INCREF(KokopopError);
    PyModule_AddObject(m, "KokopopError", KokopopError);

    Py_INCREF(&ModelType);
    PyModule_AddObject(m, "Model", reinterpret_cast<PyObject *>(&ModelType));
    Py_INCREF(&AudioType);
    PyModule_AddObject(m, "Audio", reinterpret_cast<PyObject *>(&AudioType));
    Py_INCREF(&AudioChunkType);
    PyModule_AddObject(m, "AudioChunk", reinterpret_cast<PyObject *>(&AudioChunkType));
    Py_INCREF(&SynthesisSessionType);
    PyModule_AddObject(m, "SynthesisSession", reinterpret_cast<PyObject *>(&SynthesisSessionType));
    Py_INCREF(&AudioEncoderType);
    PyModule_AddObject(m, "AudioEncoder", reinterpret_cast<PyObject *>(&AudioEncoderType));

    return m;
}
