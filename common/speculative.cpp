#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <unordered_map>
#include <utility>
#include <vector>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(
            llama_seq_id seq_id,
            uint16_t n_accepted,
            llama_token target_token,
            bool is_other,
            const std::vector<common_sampler_accept_trace> * target_trace) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];


            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, llama_token /*target_token*/, bool /*is_other*/, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = const_cast<llama_model *>(llama_get_model(ctx_dft));
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, llama_token /*target_token*/, bool /*is_other*/, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }

    bool need_embd() const override {
        return false;
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = const_cast<llama_model *>(llama_get_model(ctx_dft));
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        LOG_INF("%s: adding speculative implementation 'draft-dflash'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u\n", __func__, block_size, mask_token_id, target_layer_ids_n);

        // DFlash input is [id_last, <mask> * (block_size-1)], so it can draft at most block_size-1 tokens per step
        if (this->params.n_max > block_size - 1 || this->params.n_min > block_size - 1) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained DFlash block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, block_size - 1);
            this->params.n_max = std::min(this->params.n_max, block_size - 1);
            this->params.n_min = std::min(this->params.n_min, block_size - 1);
        }

        batch        = llama_batch_init(llama_n_batch(ctx_dft), 0,          n_seq);
        batch_inject = llama_batch_init(llama_n_batch(ctx_dft), n_embd_dec, n_seq);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention
    }

    ~common_speculative_impl_draft_dflash() override {
        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather this chunk's target features, interleaved by extract layer
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                    if (!layer) {
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                // fuse extracted features through DFlash encoder
                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    batch_inject.pos[i]       = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.n_past;

            int32_t n_draft = params.n_max;
            if (dp.n_max > 0) {
                n_draft = std::min(n_draft, dp.n_max);
            }

            const int32_t n_block_tokens = n_draft + 1; // id_last + n_draft * <mask>
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
            for (int32_t i = 1; i < n_block_tokens; ++i) {
                common_sampler_sample(smpl, ctx_dft, beg + i, true);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                if (cur_p->data[0].p < params.p_min) {
                    break;
                }

                common_sampler_accept(smpl, id, true);

                result.push_back(id);
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, llama_token /*target_token*/, bool /*is_other*/, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    struct mtp_cap_score {
        double   tokens_per_us = 0.0;
        uint32_t samples       = 0;
    };

    static constexpr int32_t MTP_ACCEPT_DUMP_TOP_K = 8;

    struct mtp_accept_attempt {
        llama_pos   pos         = 0;
        llama_token prev_token  = 0;
        llama_token draft_token = 0;
        int32_t     depth       = 0;
        int32_t     row_type    = 0; // 0=draft attempt, 1=confidence stop before drafting.
        float       p           = 0.0f;
        float       h_in_scale  = 1.0f;
        float       h_scale     = 1.0f;
        std::array<int32_t, MTP_ACCEPT_DUMP_TOP_K> candidate_ids = {};
        std::array<float,   MTP_ACCEPT_DUMP_TOP_K> candidate_ps  = {};
        std::vector<int8_t> h_in_q8;
        std::vector<int8_t> h_q8;
    };

    struct mtp_draft_cache_entry {
        llama_tokens tokens;
        std::array<float, 3> accept_ema = { 0.0f, 0.0f, 0.0f };
        uint32_t updates = 0;
        uint32_t hits = 0;
        uint64_t last_used = 0;
    };

    struct mtp_engram_entry {
        std::array<float, 3> accept_ema = { 0.0f, 0.0f, 0.0f };
        uint32_t updates = 0;
        uint32_t hits = 0;
        uint64_t last_used = 0;
    };

    struct mtp_draft_cache_active {
        bool valid = false;
        bool from_cache = false;
        bool engram_valid = false;
        uint64_t key = 0;
        uint64_t engram_key = 0;
        llama_tokens tokens;
    };

    static constexpr float    MTP_DRAFT_CACHE_ALPHA     = 0.25f;
    static constexpr float    MTP_ADAPT_ACCEPT_ALPHA    = 0.05f;
    static constexpr double   MTP_ADAPT_SCORE_ALPHA     = 0.20;
    static constexpr uint32_t MTP_ADAPT_WARMUP          = 24;
    static constexpr uint32_t MTP_ADAPT_MIN_CAP_SAMPLES = 6;
    static constexpr uint32_t MTP_ADAPT_PROBE_INTERVAL  = 96;

    std::vector<float> accept_pos_ema;
    std::vector<mtp_cap_score> cap_scores;
    std::vector<int32_t> active_cap;
    std::vector<int64_t> active_start_us;
    std::fstream mtp_dump;
    std::vector<int8_t> mtp_dump_q8;
    std::fstream mtp_accept_dump;
    std::fstream mtp_state_dump;
    std::vector<std::vector<mtp_accept_attempt>> mtp_accept_pending;
    uint64_t mtp_accept_dump_records = 0;
    uint64_t mtp_state_dump_records = 0;
    uint64_t mtp_accept_batch_id     = 0;
    bool     mtp_accept_dump_finished = false;
    bool     mtp_state_dump_finished = false;
    std::vector<llama_adapter_lora_ptr> mtp_lora_storage;
    llama_adapter_lora * mtp_lora_depth[3] = { nullptr, nullptr, nullptr };
    llama_adapter_lora * mtp_output_lora_depth[3] = { nullptr, nullptr, nullptr };
    llama_adapter_lora * mtp_state_lora_depth[3] = { nullptr, nullptr, nullptr };
    float mtp_state_lora_scale_depth[3] = { 1.0f, 1.0f, 1.0f };
    bool mtp_output_lora_configured = false;
    int32_t  mtp_lora_active_depth = -2;

    struct mtp_state_head {
        uint32_t n_embd = 0;
        uint32_t rank = 0;
        uint32_t n_vocab = 0;
        uint32_t flags = 0;
        float scale = 1.0f;
        std::vector<float> hnorm;
        std::vector<float> down_h;
        std::vector<float> up;
        std::vector<float> skip;
        std::vector<float> token_down;
        std::vector<float> h_norm;
        std::vector<float> r;
        uint64_t calls = 0;
        uint64_t steered = 0;

        bool ready() const {
            return n_embd > 0 && rank > 0 && n_vocab > 0;
        }

        bool identity_skip() const {
            return (flags & 1u) != 0;
        }
    };
    mtp_state_head mtp_state;
    std::vector<std::vector<float>> mtp_state_steered_h;

    uint64_t mtp_dump_records  = 0;
    bool     mtp_dump_finished = false;
    uint32_t accept_updates      = 0;
    int32_t  n_max_adaptive_last = -1;
    int32_t  next_probe_cap      = 1;

    std::unordered_map<uint64_t, mtp_draft_cache_entry> mtp_draft_cache;
    std::vector<mtp_draft_cache_active> mtp_draft_cache_active;
    std::unordered_map<uint64_t, mtp_engram_entry> mtp_engram_cache;
    std::vector<uint64_t> mtp_engram_current_key;
    std::vector<bool> mtp_engram_current_valid;
    std::vector<std::vector<float>> mtp_draft_input_h;
    uint64_t mtp_draft_cache_clock = 0;
    uint64_t mtp_draft_cache_queries = 0;
    uint64_t mtp_draft_cache_hits = 0;
    uint64_t mtp_draft_cache_tokens = 0;
    uint64_t mtp_engram_cache_hits = 0;

    std::vector<uint8_t> mtp_fr_allowed;
    uint64_t mtp_fr_queries = 0;
    uint64_t mtp_fr_first_allowed = 0;
    uint64_t mtp_fr_replaced = 0;
    uint64_t mtp_fr_blocked = 0;
    uint64_t mtp_fr_prompt_allowed = 0;
    std::vector<float> mtp_logit_bias;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
        init_mtp_fr_vocab();
        init_mtp_logit_bias();

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = mtp_fr_allowed.empty() ? 10 : std::max<int32_t>(10, (int32_t) this->params.mtp_fr_top_k);
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(
                            mtp_fr_allowed.empty() ? 10 : std::max<int32_t>(10, (int32_t) this->params.mtp_fr_top_k)));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        const bool dump_draft_rows = !this->params.mtp_train_dump.empty() && this->params.mtp_train_dump_source == "draft";
        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !dump_draft_rows);
        llama_set_mtp_hidden_lora_state(ctx_dft, this->params.mtp_lora_state);

        if (this->params.mtp_engram_layer_cache) {
            const int32_t n_layer_tgt = llama_model_n_layer(llama_get_model(ctx_tgt));
            if (this->params.mtp_engram_layer >= (uint32_t) n_layer_tgt) {
                throw std::runtime_error("--spec-mtp-engram-layer is outside the target model layer range");
            }
            llama_set_embeddings_layer_inp(ctx_tgt, this->params.mtp_engram_layer, true);
        }

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }

        const size_t n_cap = (size_t) std::max(1, this->params.n_max);
        accept_pos_ema.assign(n_cap, 0.75f);
        cap_scores.assign(n_cap + 1, {});
        active_cap.assign(n_seq, -1);
        active_start_us.assign(n_seq, 0);
        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        mtp_accept_pending.assign(n_seq, {});
        mtp_draft_input_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        mtp_state_steered_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        mtp_draft_cache_active.assign(n_seq, {});
        mtp_engram_current_key.assign(n_seq, 0);
        mtp_engram_current_valid.assign(n_seq, false);

        if (this->params.mtp_draft_cache) {
            SPC_INF("MTP draft cache enabled: size=%u, context=%u, min_hits=%u, min_accept=%.3f\n",
                    this->params.mtp_draft_cache_size,
                    this->params.mtp_draft_cache_context,
                    this->params.mtp_draft_cache_min_hits,
                    this->params.mtp_draft_cache_min_accept);
        }
        if (this->params.mtp_engram_layer_cache) {
            SPC_INF("MTP layer engram cache enabled: layer=%u, size=%u, min_hits=%u, min_accept=%.3f\n",
                    this->params.mtp_engram_layer,
                    this->params.mtp_engram_cache_size,
                    this->params.mtp_engram_min_hits,
                    this->params.mtp_engram_min_accept);
        }

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        init_mtp_train_dump();
        init_mtp_accept_dump();
        init_mtp_state_dump();
        init_mtp_loras();
        init_mtp_state_head();
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        set_mtp_lora_depth(-1);
        if (mtp_dump.is_open()) {
            mtp_dump.close();
        }
        if (mtp_accept_dump.is_open()) {
            mtp_accept_dump.close();
        }
        if (mtp_state_dump.is_open()) {
            mtp_state_dump.close();
        }
        mtp_lora_storage.clear();
        llama_batch_free(batch);
    }

    static uint64_t mtp_draft_cache_mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }

    void init_mtp_fr_vocab() {
        if (params.mtp_fr_vocab.empty()) {
            return;
        }

        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(params.ctx_dft));
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        mtp_fr_allowed.assign((size_t) n_vocab, 0);

        std::ifstream in(params.mtp_fr_vocab);
        if (!in.is_open()) {
            throw std::runtime_error("failed to open MTP FR vocab: " + params.mtp_fr_vocab);
        }

        uint64_t loaded = 0;
        std::string line;
        while (std::getline(in, line)) {
            const size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.resize(comment);
            }

            char * end = nullptr;
            const long value = std::strtol(line.c_str(), &end, 10);
            if (end == line.c_str()) {
                continue;
            }
            if (value < 0 || value >= n_vocab) {
                continue;
            }
            if (!mtp_fr_allowed[(size_t) value]) {
                mtp_fr_allowed[(size_t) value] = 1;
                ++loaded;
            }
        }

        if (loaded == 0) {
            throw std::runtime_error("MTP FR vocab did not contain any valid token ids: " + params.mtp_fr_vocab);
        }

        SPC_INF("MTP FR vocab enabled: path='%s', tokens=%" PRIu64 ", top_k=%u, prompt_tokens=%d, prompt_context=%u\n",
                params.mtp_fr_vocab.c_str(), loaded, params.mtp_fr_top_k,
                (int) params.mtp_fr_prompt_tokens, params.mtp_fr_prompt_context);
    }


    void init_mtp_logit_bias() {
        if (params.mtp_logit_bias.empty()) {
            return;
        }

        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(params.ctx_dft));
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        std::ifstream in(params.mtp_logit_bias);
        if (!in.is_open()) {
            throw std::runtime_error("failed to open MTP logit bias file: " + params.mtp_logit_bias);
        }

        uint64_t loaded = 0;
        std::string line;
        while (std::getline(in, line)) {
            const size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.resize(comment);
            }

            char * end = nullptr;
            const long token = std::strtol(line.c_str(), &end, 10);
            if (end == line.c_str()) {
                continue;
            }

            const float bias = std::strtof(end, &end);
            if (token < 0 || token >= n_vocab || !std::isfinite(bias) || bias == 0.0f) {
                continue;
            }

            if (mtp_logit_bias.empty()) {
                mtp_logit_bias.assign((size_t) n_vocab, 0.0f);
            }
            mtp_logit_bias[(size_t) token] = bias;
            ++loaded;
        }

        if (loaded == 0) {
            throw std::runtime_error("MTP logit bias file did not contain any valid rows: " + params.mtp_logit_bias);
        }

        SPC_INF("MTP draft logit bias enabled: path='%s', rows=%" PRIu64 "\n",
                params.mtp_logit_bias.c_str(), loaded);
    }
    bool mtp_fr_prompt_has_token(const common_speculative_draft_params & dp, llama_token id) const {
        if (!params.mtp_fr_prompt_tokens || dp.prompt == nullptr || params.mtp_fr_prompt_context == 0) {
            return false;
        }

        const llama_tokens & prompt = *dp.prompt;
        const size_t prompt_size = prompt.size();
        const size_t begin = prompt_size > params.mtp_fr_prompt_context ? prompt_size - params.mtp_fr_prompt_context : 0;
        for (size_t i = begin; i < prompt_size; ++i) {
            if (prompt[i] == id) {
                return true;
            }
        }
        return false;
    }

    bool mtp_fr_token_allowed(const common_speculative_draft_params & dp, llama_token id, bool * allowed_by_prompt) const {
        if (mtp_fr_allowed.empty()) {
            return true;
        }
        if (id >= 0 && (size_t) id < mtp_fr_allowed.size() && mtp_fr_allowed[(size_t) id]) {
            if (allowed_by_prompt) {
                *allowed_by_prompt = false;
            }
            return true;
        }
        if (mtp_fr_prompt_has_token(dp, id)) {
            if (allowed_by_prompt) {
                *allowed_by_prompt = true;
            }
            return true;
        }
        return false;
    }

    llama_token mtp_fr_select_candidate(
            const common_speculative_draft_params & dp,
            const llama_token_data_array * cur_p,
            float & p_selected) {
        if (cur_p == nullptr || cur_p->size <= 0) {
            p_selected = 0.0f;
            return LLAMA_TOKEN_NULL;
        }

        if (mtp_fr_allowed.empty() && mtp_logit_bias.empty()) {
            p_selected = cur_p->data[0].p;
            return cur_p->data[0].id;
        }

        if (!mtp_fr_allowed.empty()) {
            ++mtp_fr_queries;
        }

        llama_token best_id = LLAMA_TOKEN_NULL;
        float best_p = 0.0f;
        double best_score = -INFINITY;
        size_t best_k = 0;
        bool best_allowed_by_prompt = false;

        for (size_t k = 0; k < cur_p->size; ++k) {
            bool allowed_by_prompt = false;
            const llama_token id = cur_p->data[k].id;
            if (!mtp_fr_token_allowed(dp, id, &allowed_by_prompt)) {
                continue;
            }

            const float p = std::isfinite(cur_p->data[k].p) ? cur_p->data[k].p : 0.0f;
            double score = std::log((double) std::max(p, 1.0e-9f));
            if (id >= 0 && (size_t) id < mtp_logit_bias.size()) {
                score += mtp_logit_bias[(size_t) id];
            }

            if (best_id == LLAMA_TOKEN_NULL || score > best_score) {
                best_id = id;
                best_p = p;
                best_score = score;
                best_k = k;
                best_allowed_by_prompt = allowed_by_prompt;
            }
        }

        if (best_id == LLAMA_TOKEN_NULL) {
            if (!mtp_fr_allowed.empty()) {
                ++mtp_fr_blocked;
            }
            p_selected = 0.0f;
            return LLAMA_TOKEN_NULL;
        }

        if (!mtp_fr_allowed.empty()) {
            if (best_k == 0) {
                ++mtp_fr_first_allowed;
            } else {
                ++mtp_fr_replaced;
            }
            if (best_allowed_by_prompt) {
                ++mtp_fr_prompt_allowed;
            }
        }

        if (!mtp_logit_bias.empty()) {
            p_selected = (float) std::min(1.0, std::exp(best_score));
        } else {
            p_selected = best_p;
        }
        return best_id;
    }

    void maybe_log_mtp_fr_stats() const {
        if (mtp_fr_allowed.empty() || mtp_fr_queries == 0) {
            return;
        }
        if (mtp_fr_queries != 1 && mtp_fr_queries % 512 != 0) {
            return;
        }

        SPC_INF("MTP FR vocab stats: queries=%" PRIu64 ", first=%" PRIu64 ", replaced=%" PRIu64 ", blocked=%" PRIu64 ", prompt_allowed=%" PRIu64 "\n",
                mtp_fr_queries, mtp_fr_first_allowed, mtp_fr_replaced, mtp_fr_blocked, mtp_fr_prompt_allowed);
    }

    uint64_t mtp_engram_signature(const float * row) const {
        if (!params.mtp_engram_layer_cache || row == nullptr) {
            return 0;
        }

        float center = 0.0f;
        for (int32_t bit = 0; bit < 64; ++bit) {
            const int32_t idx = (int32_t) (((uint64_t) bit * (uint64_t) n_embd) / 64ULL);
            center += std::isfinite(row[idx]) ? row[idx] : 0.0f;
        }
        center /= 64.0f;

        uint64_t sig = 0;
        for (int32_t bit = 0; bit < 64; ++bit) {
            const int32_t idx = (int32_t) (((uint64_t) bit * 11400714819323198485ULL + (uint64_t) bit * 13ULL) % (uint64_t) n_embd);
            const float v = std::isfinite(row[idx]) ? row[idx] : 0.0f;
            if (v >= center) {
                sig |= 1ULL << bit;
            }
        }

        sig = mtp_draft_cache_mix(sig, params.mtp_engram_layer);
        return sig == 0 ? 1 : sig;
    }

    void set_mtp_engram_current(llama_seq_id seq_id, const float * row) {
        if (!params.mtp_engram_layer_cache || seq_id < 0 || seq_id >= (llama_seq_id) mtp_engram_current_key.size()) {
            return;
        }

        const uint64_t key = mtp_engram_signature(row);
        mtp_engram_current_key[seq_id] = key;
        mtp_engram_current_valid[seq_id] = key != 0;
    }

    int32_t mtp_engram_trusted_len(llama_seq_id seq_id, int32_t requested) {
        if (!params.mtp_engram_layer_cache) {
            return requested;
        }
        if (requested <= 0 || seq_id < 0 || seq_id >= (llama_seq_id) mtp_engram_current_key.size() ||
                !mtp_engram_current_valid[seq_id]) {
            return 0;
        }

        auto it = mtp_engram_cache.find(mtp_engram_current_key[seq_id]);
        if (it == mtp_engram_cache.end()) {
            return 0;
        }

        auto & entry = it->second;
        entry.last_used = ++mtp_draft_cache_clock;
        if (entry.updates < params.mtp_engram_min_hits) {
            return 0;
        }

        int32_t n = 0;
        const int32_t limit = std::min<int32_t>(requested, (int32_t) entry.accept_ema.size());
        for (; n < limit; ++n) {
            if (entry.accept_ema[(size_t) n] < params.mtp_engram_min_accept) {
                break;
            }
        }
        if (n > 0) {
            ++entry.hits;
            ++mtp_engram_cache_hits;
        }
        return n;
    }

    void evict_mtp_engram_cache_if_needed() {
        const uint32_t max_size = params.mtp_engram_cache_size;
        if (max_size == 0 || mtp_engram_cache.size() <= max_size) {
            return;
        }

        auto oldest = mtp_engram_cache.begin();
        for (auto it = mtp_engram_cache.begin(); it != mtp_engram_cache.end(); ++it) {
            if (it->second.last_used < oldest->second.last_used) {
                oldest = it;
            }
        }
        mtp_engram_cache.erase(oldest);
    }

    void update_mtp_engram_cache(uint64_t key, const llama_tokens & tokens, uint16_t n_accepted) {
        if (!params.mtp_engram_layer_cache || key == 0 || tokens.empty()) {
            return;
        }

        auto & entry = mtp_engram_cache[key];
        const uint16_t n_acc = std::min<uint16_t>(n_accepted, (uint16_t) tokens.size());
        const int32_t limit = std::min<int32_t>((int32_t) tokens.size(), (int32_t) entry.accept_ema.size());
        for (int32_t i = 0; i < limit; ++i) {
            const float sample = i < n_acc ? 1.0f : 0.0f;
            if (entry.updates == 0) {
                entry.accept_ema[(size_t) i] = sample;
            } else {
                entry.accept_ema[(size_t) i] += MTP_DRAFT_CACHE_ALPHA * (sample - entry.accept_ema[(size_t) i]);
            }
        }
        ++entry.updates;
        entry.last_used = ++mtp_draft_cache_clock;
        evict_mtp_engram_cache_if_needed();
    }

    uint64_t mtp_draft_cache_key(const common_speculative_draft_params & dp) const {
        const uint32_t ctx_n = std::max<uint32_t>(1, params.mtp_draft_cache_context);
        uint64_t h = 1469598103934665603ULL;
        h = mtp_draft_cache_mix(h, ctx_n);
        h = mtp_draft_cache_mix(h, (uint64_t) params.n_max);

        const llama_tokens * prompt = dp.prompt;
        const size_t prompt_size = prompt == nullptr ? 0 : prompt->size();
        const size_t begin = prompt_size > ctx_n ? prompt_size - ctx_n : 0;
        size_t mixed = 0;
        for (size_t i = begin; i < prompt_size; ++i) {
            h = mtp_draft_cache_mix(h, (uint64_t) (uint32_t) (*prompt)[i]);
            ++mixed;
        }

        if (prompt_size == 0 || prompt->back() != dp.id_last) {
            h = mtp_draft_cache_mix(h, (uint64_t) (uint32_t) dp.id_last);
            ++mixed;
        }

        h = mtp_draft_cache_mix(h, mixed);
        return h == 0 ? 1 : h;
    }

    int32_t mtp_draft_cache_limit_for(const common_speculative_draft_params & dp, int32_t n_max_eff) const {
        int32_t limit = std::max(0, n_max_eff);
        if (dp.n_max > 0) {
            limit = std::min(limit, dp.n_max);
        }
        return std::min(limit, params.n_max);
    }

    int32_t mtp_draft_cache_trusted_len(
            const mtp_draft_cache_entry & entry,
            const common_speculative_draft_params & dp,
            int32_t n_max_eff) const {
        if (!params.mtp_draft_cache || entry.updates < params.mtp_draft_cache_min_hits) {
            return 0;
        }

        const int32_t limit = std::min<int32_t>((int32_t) entry.tokens.size(), mtp_draft_cache_limit_for(dp, n_max_eff));
        int32_t n = 0;
        for (; n < limit && n < (int32_t) entry.accept_ema.size(); ++n) {
            if (entry.accept_ema[(size_t) n] < params.mtp_draft_cache_min_accept) {
                break;
            }
        }
        return n;
    }

    bool try_mtp_draft_cache(llama_seq_id seq_id, common_speculative_draft_params & dp, int32_t n_max_eff) {
        if (!params.mtp_draft_cache || params.mtp_draft_cache_size == 0 || seq_id < 0 ||
                seq_id >= (llama_seq_id) mtp_draft_cache_active.size()) {
            return false;
        }

        ++mtp_draft_cache_queries;
        const uint64_t key = mtp_draft_cache_key(dp);
        auto it = mtp_draft_cache.find(key);
        if (it == mtp_draft_cache.end()) {
            return false;
        }

        auto & entry = it->second;
        int32_t n = mtp_draft_cache_trusted_len(entry, dp, n_max_eff);
        n = mtp_engram_trusted_len(seq_id, n);
        entry.last_used = ++mtp_draft_cache_clock;
        if (n <= 0) {
            return false;
        }

        auto & result = *dp.result;
        result.insert(result.end(), entry.tokens.begin(), entry.tokens.begin() + n);
        ++entry.hits;
        ++mtp_draft_cache_hits;
        mtp_draft_cache_tokens += (uint64_t) n;

        auto & active = mtp_draft_cache_active[seq_id];
        active.valid = true;
        active.from_cache = true;
        active.key = key;
        if (params.mtp_engram_layer_cache && seq_id >= 0 && seq_id < (llama_seq_id) mtp_engram_current_key.size()) {
            active.engram_valid = mtp_engram_current_valid[seq_id];
            active.engram_key = mtp_engram_current_key[seq_id];
        }
        active.tokens.assign(result.begin(), result.end());

        if (mtp_draft_cache_hits == 1 || mtp_draft_cache_hits % 256 == 0) {
            SPC_INF("MTP draft cache hit: hits=%" PRIu64 ", queries=%" PRIu64 ", tokens=%" PRIu64 ", size=%zu, reused=%d\n",
                    mtp_draft_cache_hits, mtp_draft_cache_queries, mtp_draft_cache_tokens, mtp_draft_cache.size(), n);
        }

        return true;
    }

    void evict_mtp_draft_cache_if_needed() {
        const uint32_t max_size = params.mtp_draft_cache_size;
        if (max_size == 0 || mtp_draft_cache.size() <= max_size) {
            return;
        }

        auto oldest = mtp_draft_cache.begin();
        for (auto it = mtp_draft_cache.begin(); it != mtp_draft_cache.end(); ++it) {
            if (it->second.last_used < oldest->second.last_used) {
                oldest = it;
            }
        }
        mtp_draft_cache.erase(oldest);
    }

    void remember_mtp_draft_cache_attempt(llama_seq_id seq_id, const common_speculative_draft_params & dp, bool from_cache) {
        if (!params.mtp_draft_cache || seq_id < 0 || seq_id >= (llama_seq_id) mtp_draft_cache_active.size() || dp.result == nullptr) {
            return;
        }

        auto & active = mtp_draft_cache_active[seq_id];
        if (active.valid && active.from_cache) {
            return;
        }

        active.valid = true;
        active.from_cache = from_cache;
        active.key = mtp_draft_cache_key(dp);
        if (params.mtp_engram_layer_cache && seq_id >= 0 && seq_id < (llama_seq_id) mtp_engram_current_key.size()) {
            active.engram_valid = mtp_engram_current_valid[seq_id];
            active.engram_key = mtp_engram_current_key[seq_id];
        }
        active.tokens.assign(dp.result->begin(), dp.result->end());
    }

    void update_mtp_draft_cache(llama_seq_id seq_id, uint16_t n_accepted) {
        if (!params.mtp_draft_cache || seq_id < 0 || seq_id >= (llama_seq_id) mtp_draft_cache_active.size()) {
            return;
        }

        auto & active = mtp_draft_cache_active[seq_id];
        if (!active.valid || active.tokens.empty()) {
            return;
        }

        auto & entry = mtp_draft_cache[active.key];
        if (entry.tokens != active.tokens) {
            entry.tokens = active.tokens;
            entry.accept_ema = { 0.0f, 0.0f, 0.0f };
            entry.updates = 0;
            entry.hits = 0;
        }

        const uint16_t n_acc = std::min<uint16_t>(n_accepted, (uint16_t) active.tokens.size());
        const int32_t limit = std::min<int32_t>((int32_t) active.tokens.size(), (int32_t) entry.accept_ema.size());
        for (int32_t i = 0; i < limit; ++i) {
            const float sample = i < n_acc ? 1.0f : 0.0f;
            if (entry.updates == 0) {
                entry.accept_ema[(size_t) i] = sample;
            } else {
                entry.accept_ema[(size_t) i] += MTP_DRAFT_CACHE_ALPHA * (sample - entry.accept_ema[(size_t) i]);
            }
        }
        ++entry.updates;
        entry.last_used = ++mtp_draft_cache_clock;
        if (active.engram_valid) {
            update_mtp_engram_cache(active.engram_key, active.tokens, n_accepted);
        }
        evict_mtp_draft_cache_if_needed();

        active = {};
    }

    template <typename T>
    void mtp_dump_write_scalar(const T & value) {
        mtp_dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    void init_mtp_loras() {
        auto * ctx_dft = params.ctx_dft;
        if (ctx_dft == nullptr) {
            return;
        }

        llama_model * model_dft = const_cast<llama_model *>(llama_get_model(ctx_dft));
        for (int i = 0; i < 3; ++i) {
            const std::string & path = params.mtp_lora_depth[i];
            if (!path.empty()) {
                llama_adapter_lora_ptr lora;
                lora.reset(llama_adapter_lora_init(model_dft, path.c_str()));
                if (lora == nullptr) {
                    throw std::runtime_error("failed to load MTP draft LoRA adapter: " + path);
                }

                mtp_lora_depth[i] = lora.get();
                mtp_lora_storage.emplace_back(std::move(lora));
                SPC_INF("MTP draft LoRA depth %d enabled: path='%s'\n", i + 1, path.c_str());
            }

            const std::string & output_path = params.mtp_output_lora_depth[i];
            if (!output_path.empty()) {
                llama_adapter_lora_ptr lora;
                lora.reset(llama_adapter_lora_init(model_dft, output_path.c_str()));
                if (lora == nullptr) {
                    throw std::runtime_error("failed to load MTP output LoRA adapter: " + output_path);
                }

                mtp_output_lora_depth[i] = lora.get();
                mtp_output_lora_configured = true;
                mtp_lora_storage.emplace_back(std::move(lora));
                SPC_INF("MTP output LoRA depth %d enabled: path='%s'\n", i + 1, output_path.c_str());
            }

            const std::string & state_path = params.mtp_state_lora_depth[i];
            if (!state_path.empty()) {
                llama_adapter_lora_ptr lora;
                lora.reset(llama_adapter_lora_init(model_dft, state_path.c_str()));
                if (lora == nullptr) {
                    throw std::runtime_error("failed to load MTP state LoRA adapter: " + state_path);
                }

                mtp_state_lora_depth[i] = lora.get();
                mtp_state_lora_scale_depth[i] = params.mtp_state_lora_scale_depth[i];
                mtp_lora_storage.emplace_back(std::move(lora));
                SPC_INF("MTP state LoRA depth %d enabled: path='%s', scale=%.3f\n",
                        i + 1, state_path.c_str(), mtp_state_lora_scale_depth[i]);
            }
        }
    }

    void set_mtp_lora_depth(int32_t depth) {
        if (mtp_lora_active_depth == depth) {
            return;
        }

        auto * ctx_dft = params.ctx_dft;
        if (ctx_dft == nullptr) {
            return;
        }

        if (depth >= 0 && depth < 3 && mtp_lora_depth[depth] != nullptr) {
            llama_set_mtp_hidden_lora(ctx_dft, mtp_lora_depth[depth], 1.0f);
        } else {
            llama_set_mtp_hidden_lora(ctx_dft, nullptr, 1.0f);
        }
        if (mtp_output_lora_configured) {
            if (depth >= 0 && depth < 3 && mtp_output_lora_depth[depth] != nullptr) {
                llama_adapter_lora * adapters[] = { mtp_output_lora_depth[depth] };
                float scales[] = { 1.0f };
                llama_set_adapters_lora(ctx_dft, adapters, 1, scales);
            } else {
                llama_set_adapters_lora(ctx_dft, nullptr, 0, nullptr);
            }
        }
        if (depth >= 0 && depth < 3 && mtp_state_lora_depth[depth] != nullptr) {
            llama_set_mtp_hidden_state_lora(ctx_dft, mtp_state_lora_depth[depth], mtp_state_lora_scale_depth[depth]);
        } else {
            llama_set_mtp_hidden_state_lora(ctx_dft, nullptr, 1.0f);
        }
        mtp_lora_active_depth = depth;
    }

    template <typename T>
    static bool mtp_state_read_scalar(std::istream & in, T & value) {
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        return in.good();
    }

    static bool mtp_state_read_f32(std::istream & in, std::vector<float> & dst, size_t n) {
        dst.resize(n);
        in.read(reinterpret_cast<char *>(dst.data()), (std::streamsize) (n * sizeof(float)));
        return in.good();
    }

    void init_mtp_state_head() {
        if (params.mtp_state_head.empty()) {
            return;
        }

        std::ifstream in(params.mtp_state_head, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("failed to open MTP direct-state head: " + params.mtp_state_head);
        }

        char magic[8] = {};
        uint32_t version = 0;
        uint32_t file_n_embd = 0;
        uint32_t rank = 0;
        uint32_t n_vocab = 0;
        uint32_t flags = 0;
        float model_scale = 0.0f;
        in.read(magic, sizeof(magic));
        if (!in.good() ||
                memcmp(magic, "MTPDSH1", 8) != 0 ||
                !mtp_state_read_scalar(in, version) ||
                !mtp_state_read_scalar(in, file_n_embd) ||
                !mtp_state_read_scalar(in, rank) ||
                !mtp_state_read_scalar(in, n_vocab) ||
                !mtp_state_read_scalar(in, flags) ||
                !mtp_state_read_scalar(in, model_scale)) {
            throw std::runtime_error("invalid MTP direct-state head header: " + params.mtp_state_head);
        }
        if (version != 1 || file_n_embd != (uint32_t) n_embd || rank == 0 || n_vocab == 0 || (flags & ~1u) != 0) {
            throw std::runtime_error("unsupported MTP direct-state head shape: " + params.mtp_state_head);
        }

        mtp_state.n_embd = file_n_embd;
        mtp_state.rank = rank;
        mtp_state.n_vocab = n_vocab;
        mtp_state.flags = flags;
        mtp_state.scale = model_scale;
        if (!mtp_state_read_f32(in, mtp_state.hnorm, n_embd) ||
                !mtp_state_read_f32(in, mtp_state.down_h, (size_t) rank * n_embd) ||
                !mtp_state_read_f32(in, mtp_state.up, (size_t) n_embd * rank)) {
            throw std::runtime_error("truncated MTP direct-state head: " + params.mtp_state_head);
        }
        if (!mtp_state.identity_skip() && !mtp_state_read_f32(in, mtp_state.skip, (size_t) n_embd * n_embd)) {
            throw std::runtime_error("truncated MTP direct-state head skip matrix: " + params.mtp_state_head);
        }
        if (!mtp_state_read_f32(in, mtp_state.token_down, (size_t) n_vocab * rank)) {
            throw std::runtime_error("truncated MTP direct-state head token table: " + params.mtp_state_head);
        }
        mtp_state.h_norm.assign(n_embd, 0.0f);
        mtp_state.r.assign(rank, 0.0f);

        SPC_INF("MTP direct-state head enabled: path='%s', n_embd=%u, rank=%u, vocab=%u, scale=%.3f, blend=%.3f, identity_skip=%d\n",
                params.mtp_state_head.c_str(), file_n_embd, rank, n_vocab,
                (double) model_scale, (double) params.mtp_state_head_scale, (int) mtp_state.identity_skip());
    }

    bool apply_mtp_state_head(llama_token token, const float * h_in, const float * h_base, float * h_out) {
        if (!mtp_state.ready() || h_in == nullptr || h_base == nullptr || h_out == nullptr) {
            return false;
        }
        const float blend = std::max(0.0f, std::min(1.0f, params.mtp_state_head_scale));
        if (blend <= 0.0f) {
            return false;
        }
        ++mtp_state.calls;
        if (token < 0 || (uint32_t) token >= mtp_state.n_vocab) {
            return false;
        }

        const uint32_t n = mtp_state.n_embd;
        const uint32_t rnk = mtp_state.rank;
        float mean_sq = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            const float v = std::isfinite(h_in[i]) ? h_in[i] : 0.0f;
            mean_sq += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(mean_sq / (float) n + 1.0e-6f);
        for (uint32_t i = 0; i < n; ++i) {
            mtp_state.h_norm[i] = h_in[i] * inv_rms * mtp_state.hnorm[i];
        }

        const float * token_r = mtp_state.token_down.data() + (size_t) token * rnk;
        for (uint32_t r = 0; r < rnk; ++r) {
            const float * w = mtp_state.down_h.data() + (size_t) r * n;
            float acc = token_r[r];
            for (uint32_t i = 0; i < n; ++i) {
                acc += w[i] * mtp_state.h_norm[i];
            }
            mtp_state.r[r] = acc;
        }

        const float keep = 1.0f - blend;
        for (uint32_t i = 0; i < n; ++i) {
            const float * up = mtp_state.up.data() + (size_t) i * rnk;
            float pred = mtp_state.h_norm[i];
            if (!mtp_state.identity_skip()) {
                const float * skip = mtp_state.skip.data() + (size_t) i * n;
                pred = 0.0f;
                for (uint32_t j = 0; j < n; ++j) {
                    pred += skip[j] * mtp_state.h_norm[j];
                }
            }
            float delta = 0.0f;
            for (uint32_t r = 0; r < rnk; ++r) {
                delta += up[r] * mtp_state.r[r];
            }
            pred += delta * mtp_state.scale;
            h_out[i] = keep * h_base[i] + blend * pred;
        }

        ++mtp_state.steered;
        if (mtp_state.steered == 1 || mtp_state.steered % 512 == 0) {
            SPC_INF("MTP direct-state head steered rows=%" PRIu64 " calls=%" PRIu64 "\n",
                    mtp_state.steered, mtp_state.calls);
        }
        return true;
    }

    template <typename T>
    static bool mtp_dump_read_scalar(std::istream & in, T & value) {
        in.read(reinterpret_cast<char *>(&value), sizeof(value));
        return in.good();
    }

    bool init_mtp_train_dump_append(
            const char expected_magic[8],
            uint32_t expected_version,
            uint32_t expected_format,
            uint32_t expected_labels,
            uint32_t expected_meta) {
        std::ifstream in(params.mtp_train_dump, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            return false;
        }

        const std::streamoff file_size = in.tellg();
        if (file_size < 0) {
            SPC_WRN("failed to inspect existing MTP training dump '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump_finished = true;
            return true;
        }
        if (file_size == 0) {
            return false;
        }

        static constexpr std::streamoff header_size = 36;
        if (file_size < header_size) {
            SPC_WRN("refusing to append to short MTP training dump '%s' (size=%" PRId64 ")\n",
                    params.mtp_train_dump.c_str(), (int64_t) file_size);
            mtp_dump_finished = true;
            return true;
        }

        in.seekg(0, std::ios::beg);
        char magic[8] = {};
        uint32_t version = 0;
        uint32_t file_n_embd = 0;
        uint32_t n_labels = 0;
        uint32_t format = 0;
        uint32_t meta = 0;
        uint64_t old_limit = 0;
        in.read(magic, sizeof(magic));
        if (!in.good() ||
                !mtp_dump_read_scalar(in, version) ||
                !mtp_dump_read_scalar(in, file_n_embd) ||
                !mtp_dump_read_scalar(in, n_labels) ||
                !mtp_dump_read_scalar(in, format) ||
                !mtp_dump_read_scalar(in, meta) ||
                !mtp_dump_read_scalar(in, old_limit)) {
            SPC_WRN("failed to read existing MTP training dump header '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump_finished = true;
            return true;
        }

        if (memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
                version != expected_version ||
                file_n_embd != (uint32_t) n_embd ||
                n_labels != expected_labels ||
                format != expected_format ||
                meta != expected_meta) {
            SPC_WRN("refusing to append incompatible MTP training dump '%s' "
                    "(version=%u, n_embd=%u, labels=%u, format=%u, meta=%u)\n",
                    params.mtp_train_dump.c_str(), version, file_n_embd, n_labels, format, meta);
            mtp_dump_finished = true;
            return true;
        }

        const uint64_t record_size = (uint64_t) expected_meta + (uint64_t) n_embd;
        const uint64_t bytes = (uint64_t) (file_size - header_size);
        if (bytes % record_size != 0) {
            SPC_WRN("refusing to append MTP training dump '%s' with partial trailing record (%" PRIu64 " trailing bytes)\n",
                    params.mtp_train_dump.c_str(), bytes % record_size);
            mtp_dump_finished = true;
            return true;
        }

        mtp_dump_records = bytes / record_size;
        if (params.mtp_train_dump_limit > 0 && mtp_dump_records >= params.mtp_train_dump_limit) {
            SPC_INF("MTP training dump already reached limit: records=%" PRIu64 ", limit=%" PRIu64 ", path='%s'\n",
                    mtp_dump_records, params.mtp_train_dump_limit, params.mtp_train_dump.c_str());
            mtp_dump_finished = true;
            return true;
        }

        mtp_dump.open(params.mtp_train_dump, std::ios::binary | std::ios::in | std::ios::out);
        if (!mtp_dump.is_open()) {
            SPC_WRN("failed to open MTP training dump for append '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump_finished = true;
            return true;
        }

        mtp_dump.seekp(28, std::ios::beg);
        mtp_dump_write_scalar(params.mtp_train_dump_limit);
        mtp_dump.seekp(0, std::ios::end);
        if (!mtp_dump.good()) {
            SPC_WRN("failed to prepare MTP training dump append '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump.close();
            mtp_dump_finished = true;
            return true;
        }

        SPC_INF("MTP training dump append enabled: path='%s', source=%s, n_embd=%d, format=q8_row_scale, existing=%" PRIu64 ", old_limit=%" PRIu64 ", limit=%" PRIu64 "\n",
                params.mtp_train_dump.c_str(), params.mtp_train_dump_source.c_str(), n_embd, mtp_dump_records, old_limit, params.mtp_train_dump_limit);
        return true;
    }

    void init_mtp_train_dump() {
        if (params.mtp_train_dump.empty()) {
            return;
        }

        mtp_dump_q8.resize((size_t) n_embd);

        const char magic[8] = { 'M', 'T', 'P', 'D', 'M', 'P', '1', '\0' };
        const uint32_t version             = 1;
        const uint32_t format_q8_row_scale = 1;
        const uint32_t n_labels            = 3;
        const uint32_t record_meta_bytes   = 28; // seq_id, pos, token, 3 labels, f32 scale

        if (params.mtp_train_dump_append &&
                init_mtp_train_dump_append(magic, version, format_q8_row_scale, n_labels, record_meta_bytes)) {
            return;
        }

        mtp_dump.open(params.mtp_train_dump, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!mtp_dump.is_open()) {
            SPC_WRN("failed to open MTP training dump '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump_finished = true;
            return;
        }

        mtp_dump.write(magic, sizeof(magic));
        mtp_dump_write_scalar(version);
        mtp_dump_write_scalar((uint32_t) n_embd);
        mtp_dump_write_scalar(n_labels);
        mtp_dump_write_scalar(format_q8_row_scale);
        mtp_dump_write_scalar(record_meta_bytes);
        mtp_dump_write_scalar(params.mtp_train_dump_limit);

        if (!mtp_dump.good()) {
            SPC_WRN("failed to write MTP training dump header '%s'\n", params.mtp_train_dump.c_str());
            mtp_dump.close();
            mtp_dump_finished = true;
            return;
        }

        SPC_INF("MTP training dump enabled: path='%s', source=%s, n_embd=%d, format=q8_row_scale, limit=%" PRIu64 "\n",
                params.mtp_train_dump.c_str(), params.mtp_train_dump_source.c_str(), n_embd, params.mtp_train_dump_limit);
    }

    void finish_mtp_train_dump_if_needed() {
        if (!mtp_dump.is_open() || mtp_dump_finished) {
            return;
        }
        if (params.mtp_train_dump_limit > 0 && mtp_dump_records >= params.mtp_train_dump_limit) {
            SPC_INF("MTP training dump reached limit: records=%" PRIu64 ", path='%s'\n",
                    mtp_dump_records, params.mtp_train_dump.c_str());
            mtp_dump.close();
            mtp_dump_finished = true;
        }
    }

    template <typename T>
    void mtp_accept_dump_write_scalar(const T & value) {
        mtp_accept_dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    template <typename T>
    void mtp_state_dump_write_scalar(const T & value) {
        mtp_state_dump.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    static constexpr uint32_t mtp_accept_dump_header_bytes = 32;
    static constexpr uint32_t mtp_accept_dump_meta_bytes = 188;
    static constexpr uint32_t mtp_accept_dump_format_q8_row_scale = 1;
    static constexpr uint32_t mtp_state_dump_meta_bytes = 56;
    static constexpr uint32_t mtp_state_dump_format_q8_pair_scale = 1;

    uint32_t mtp_accept_dump_record_bytes() const {
        return mtp_accept_dump_meta_bytes + (uint32_t) n_embd;
    }

    uint32_t mtp_state_dump_record_bytes() const {
        return mtp_state_dump_meta_bytes + 2u * (uint32_t) n_embd;
    }

    bool init_mtp_accept_dump_append(const char expected_magic[8], uint32_t expected_version) {
        std::ifstream in(params.mtp_accept_dump, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            return false;
        }

        const std::streamoff file_size = in.tellg();
        if (file_size < 0) {
            SPC_WRN("failed to inspect existing MTP accept/reject dump '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump_finished = true;
            return true;
        }
        if (file_size == 0) {
            return false;
        }
        if (file_size < mtp_accept_dump_header_bytes) {
            SPC_WRN("refusing to append to short MTP accept/reject dump '%s' (size=%" PRId64 ")\n",
                    params.mtp_accept_dump.c_str(), (int64_t) file_size);
            mtp_accept_dump_finished = true;
            return true;
        }

        in.seekg(0, std::ios::beg);
        char magic[8] = {};
        uint32_t version = 0;
        uint32_t file_n_embd = 0;
        uint32_t meta_bytes = 0;
        uint32_t format = 0;
        uint64_t old_limit = 0;
        in.read(magic, sizeof(magic));
        if (!in.good() ||
                !mtp_dump_read_scalar(in, version) ||
                !mtp_dump_read_scalar(in, file_n_embd) ||
                !mtp_dump_read_scalar(in, meta_bytes) ||
                !mtp_dump_read_scalar(in, format) ||
                !mtp_dump_read_scalar(in, old_limit)) {
            SPC_WRN("failed to read existing MTP accept/reject dump header '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump_finished = true;
            return true;
        }

        if (memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
                version != expected_version ||
                file_n_embd != (uint32_t) n_embd ||
                meta_bytes != mtp_accept_dump_meta_bytes ||
                format != mtp_accept_dump_format_q8_row_scale) {
            SPC_WRN("refusing to append incompatible MTP accept/reject dump '%s' "
                    "(version=%u, n_embd=%u, meta=%u, format=%u)\n",
                    params.mtp_accept_dump.c_str(), version, file_n_embd, meta_bytes, format);
            mtp_accept_dump_finished = true;
            return true;
        }

        const uint32_t record_bytes = mtp_accept_dump_record_bytes();
        const uint64_t bytes = (uint64_t) (file_size - mtp_accept_dump_header_bytes);
        if (bytes % record_bytes != 0) {
            SPC_WRN("refusing to append MTP accept/reject dump '%s' with partial trailing record (%" PRIu64 " trailing bytes)\n",
                    params.mtp_accept_dump.c_str(), bytes % record_bytes);
            mtp_accept_dump_finished = true;
            return true;
        }

        mtp_accept_dump_records = bytes / record_bytes;
        if (params.mtp_accept_dump_limit > 0 && mtp_accept_dump_records >= params.mtp_accept_dump_limit) {
            SPC_INF("MTP accept/reject dump already reached limit: records=%" PRIu64 ", limit=%" PRIu64 ", path='%s'\n",
                    mtp_accept_dump_records, params.mtp_accept_dump_limit, params.mtp_accept_dump.c_str());
            mtp_accept_dump_finished = true;
            return true;
        }

        mtp_accept_dump.open(params.mtp_accept_dump, std::ios::binary | std::ios::in | std::ios::out);
        if (!mtp_accept_dump.is_open()) {
            SPC_WRN("failed to open MTP accept/reject dump for append '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump_finished = true;
            return true;
        }

        mtp_accept_dump.seekp(24, std::ios::beg);
        mtp_accept_dump_write_scalar(params.mtp_accept_dump_limit);
        mtp_accept_dump.seekp(0, std::ios::end);
        if (!mtp_accept_dump.good()) {
            SPC_WRN("failed to prepare MTP accept/reject dump append '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump.close();
            mtp_accept_dump_finished = true;
            return true;
        }

        SPC_INF("MTP accept/reject dump append enabled: path='%s', n_embd=%d, existing=%" PRIu64 ", old_limit=%" PRIu64 ", limit=%" PRIu64 "\n",
                params.mtp_accept_dump.c_str(), n_embd, mtp_accept_dump_records, old_limit, params.mtp_accept_dump_limit);
        return true;
    }

    void init_mtp_accept_dump() {
        if (params.mtp_accept_dump.empty()) {
            return;
        }

        const char magic[8] = { 'M', 'T', 'P', 'A', 'C', 'C', '6', '\0' };
        const uint32_t version = 6;

        if (params.mtp_accept_dump_append && init_mtp_accept_dump_append(magic, version)) {
            return;
        }

        mtp_accept_dump.open(params.mtp_accept_dump, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!mtp_accept_dump.is_open()) {
            SPC_WRN("failed to open MTP accept/reject dump '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump_finished = true;
            return;
        }

        mtp_accept_dump.write(magic, sizeof(magic));
        mtp_accept_dump_write_scalar(version);
        mtp_accept_dump_write_scalar((uint32_t) n_embd);
        mtp_accept_dump_write_scalar(mtp_accept_dump_meta_bytes);
        mtp_accept_dump_write_scalar(mtp_accept_dump_format_q8_row_scale);
        mtp_accept_dump_write_scalar(params.mtp_accept_dump_limit);

        if (!mtp_accept_dump.good()) {
            SPC_WRN("failed to write MTP accept/reject dump header '%s'\n", params.mtp_accept_dump.c_str());
            mtp_accept_dump.close();
            mtp_accept_dump_finished = true;
            return;
        }

        SPC_INF("MTP accept/reject dump enabled: path='%s', n_embd=%d, record_size=%u, limit=%" PRIu64 "\n",
                params.mtp_accept_dump.c_str(), n_embd, mtp_accept_dump_record_bytes(), params.mtp_accept_dump_limit);
    }

    void finish_mtp_accept_dump_if_needed() {
        if (!mtp_accept_dump.is_open() || mtp_accept_dump_finished) {
            return;
        }
        if (params.mtp_accept_dump_limit > 0 && mtp_accept_dump_records >= params.mtp_accept_dump_limit) {
            SPC_INF("MTP accept/reject dump reached limit: records=%" PRIu64 ", path='%s'\n",
                    mtp_accept_dump_records, params.mtp_accept_dump.c_str());
            mtp_accept_dump.close();
            mtp_accept_dump_finished = true;
        }
    }

    float quantize_hidden_row_q8(const float * h, std::vector<int8_t> & q8) const {
        q8.resize((size_t) n_embd);

        if (h == nullptr) {
            std::fill(q8.begin(), q8.end(), 0);
            return 1.0f;
        }

        float max_abs = 0.0f;
        for (int32_t i = 0; i < n_embd; ++i) {
            max_abs = std::max(max_abs, std::fabs(h[i]));
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        for (int32_t i = 0; i < n_embd; ++i) {
            const float scaled = h[i] / scale;
            const int q = std::max(-127, std::min(127, (int) std::lround(scaled)));
            q8[(size_t) i] = (int8_t) q;
        }
        return scale;
    }

    void init_mtp_state_dump() {
        if (params.mtp_state_dump.empty()) {
            return;
        }

        const char magic[8] = { 'M', 'T', 'P', 'S', 'T', '3', '\0', '\0' };
        const uint32_t version = 1;

        mtp_state_dump.open(params.mtp_state_dump, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!mtp_state_dump.is_open()) {
            SPC_WRN("failed to open MTP transition state dump '%s'\n", params.mtp_state_dump.c_str());
            mtp_state_dump_finished = true;
            return;
        }

        mtp_state_dump.write(magic, sizeof(magic));
        mtp_state_dump_write_scalar(version);
        mtp_state_dump_write_scalar((uint32_t) n_embd);
        mtp_state_dump_write_scalar(mtp_state_dump_meta_bytes);
        mtp_state_dump_write_scalar(mtp_state_dump_format_q8_pair_scale);
        mtp_state_dump_write_scalar(params.mtp_state_dump_limit);

        if (!mtp_state_dump.good()) {
            SPC_WRN("failed to write MTP transition state dump header '%s'\n", params.mtp_state_dump.c_str());
            mtp_state_dump.close();
            mtp_state_dump_finished = true;
            return;
        }

        SPC_INF("MTP transition state dump enabled: path='%s', n_embd=%d, record_size=%u, limit=%" PRIu64 "\n",
                params.mtp_state_dump.c_str(), n_embd, mtp_state_dump_record_bytes(), params.mtp_state_dump_limit);
    }

    void finish_mtp_state_dump_if_needed() {
        if (!mtp_state_dump.is_open() || mtp_state_dump_finished) {
            return;
        }
        if (params.mtp_state_dump_limit > 0 && mtp_state_dump_records >= params.mtp_state_dump_limit) {
            SPC_INF("MTP transition state dump reached limit: records=%" PRIu64 ", path='%s'\n",
                    mtp_state_dump_records, params.mtp_state_dump.c_str());
            mtp_state_dump.close();
            mtp_state_dump_finished = true;
        }
    }

    void fill_mtp_accept_candidates(mtp_accept_attempt & attempt, const llama_token_data_array * candidates) const {
        attempt.candidate_ids.fill((int32_t) LLAMA_TOKEN_NULL);
        attempt.candidate_ps.fill(0.0f);
        if (candidates == nullptr) {
            return;
        }

        const int32_t n = std::min<int32_t>(MTP_ACCEPT_DUMP_TOP_K, (int32_t) candidates->size);
        for (int32_t i = 0; i < n; ++i) {
            attempt.candidate_ids[(size_t) i] = (int32_t) candidates->data[i].id;
            attempt.candidate_ps[(size_t) i] = std::isfinite(candidates->data[i].p) ? candidates->data[i].p : 0.0f;
        }
    }

    void add_mtp_accept_attempt(
            llama_seq_id seq_id,
            llama_pos pos,
            llama_token prev_token,
            llama_token draft_token,
            int32_t depth,
            float p,
            const float * h_in,
            const float * h,
            const llama_token_data_array * candidates,
            int32_t row_type = 0) {
        if (((!mtp_accept_dump.is_open() || mtp_accept_dump_finished) &&
                    (!mtp_state_dump.is_open() || mtp_state_dump_finished)) || h == nullptr ||
                seq_id < 0 || seq_id >= (llama_seq_id) mtp_accept_pending.size()) {
            return;
        }

        mtp_accept_attempt attempt;
        attempt.pos = pos;
        attempt.prev_token = prev_token;
        attempt.draft_token = draft_token;
        attempt.depth = depth;
        attempt.row_type = row_type;
        attempt.p = p;
        fill_mtp_accept_candidates(attempt, candidates);
        attempt.h_in_scale = quantize_hidden_row_q8(h_in, attempt.h_in_q8);
        attempt.h_scale = quantize_hidden_row_q8(h, attempt.h_q8);

        mtp_accept_pending[seq_id].push_back(std::move(attempt));
    }

    void dump_mtp_accept_records(
            llama_seq_id seq_id,
            uint16_t n_accepted,
            llama_token target_token,
            const std::vector<common_sampler_accept_trace> * target_trace) {
        if (((!mtp_accept_dump.is_open() || mtp_accept_dump_finished) &&
                    (!mtp_state_dump.is_open() || mtp_state_dump_finished)) ||
                seq_id < 0 || seq_id >= (llama_seq_id) mtp_accept_pending.size()) {
            return;
        }

        auto & attempts = mtp_accept_pending[seq_id];
        int32_t n_drafted = 0;
        for (const auto & a : attempts) {
            if (a.row_type == 0) {
                ++n_drafted;
            }
        }
        const int32_t n_acc = std::min<int32_t>(n_accepted, n_drafted);
        const uint64_t batch_id = mtp_accept_batch_id++;
        int32_t draft_idx = 0;

        for (int32_t i = 0; i < (int32_t) attempts.size(); ++i) {
            finish_mtp_accept_dump_if_needed();

            const auto & a = attempts[i];
            const bool is_draft = a.row_type == 0;
            const int32_t accepted = is_draft && draft_idx < n_acc ? 1 : 0;
            const int32_t verified = is_draft && draft_idx <= n_acc ? 1 : 0; // after the first rejection, later chain tokens were never target-verified.
            const int32_t row_target = accepted ? (int32_t) a.draft_token : (verified ? (int32_t) target_token : -1);
            const float prob = std::isfinite(a.p) ? a.p : 0.0f;
            std::array<int32_t, MTP_ACCEPT_DUMP_TOP_K> target_candidate_ids = {};
            std::array<float,   MTP_ACCEPT_DUMP_TOP_K> target_candidate_ps  = {};
            target_candidate_ids.fill((int32_t) LLAMA_TOKEN_NULL);
            target_candidate_ps.fill(0.0f);
            if (target_trace != nullptr && verified && draft_idx >= 0 && draft_idx < (int32_t) target_trace->size()) {
                const auto & trace = target_trace->at((size_t) draft_idx);
                for (int32_t k = 0; k < MTP_ACCEPT_DUMP_TOP_K; ++k) {
                    target_candidate_ids[(size_t) k] = (int32_t) trace.candidate_ids[(size_t) k];
                    target_candidate_ps[(size_t) k] = trace.candidate_ps[(size_t) k];
                }
            }

            if (mtp_accept_dump.is_open() && !mtp_accept_dump_finished) {
                mtp_accept_dump_write_scalar(batch_id);
                mtp_accept_dump_write_scalar((int32_t) seq_id);
                mtp_accept_dump_write_scalar((int32_t) a.pos);
                mtp_accept_dump_write_scalar((int32_t) a.depth);
                mtp_accept_dump_write_scalar((int32_t) a.prev_token);
                mtp_accept_dump_write_scalar((int32_t) a.draft_token);
                mtp_accept_dump_write_scalar(prob);
                mtp_accept_dump_write_scalar(accepted);
                mtp_accept_dump_write_scalar(verified);
                mtp_accept_dump_write_scalar(n_acc);
                mtp_accept_dump_write_scalar(n_drafted);
                mtp_accept_dump_write_scalar(row_target);
                mtp_accept_dump_write_scalar(a.row_type);
                for (int32_t k = 0; k < MTP_ACCEPT_DUMP_TOP_K; ++k) {
                    mtp_accept_dump_write_scalar(a.candidate_ids[(size_t) k]);
                }
                for (int32_t k = 0; k < MTP_ACCEPT_DUMP_TOP_K; ++k) {
                    mtp_accept_dump_write_scalar(a.candidate_ps[(size_t) k]);
                }
                for (int32_t k = 0; k < MTP_ACCEPT_DUMP_TOP_K; ++k) {
                    mtp_accept_dump_write_scalar(target_candidate_ids[(size_t) k]);
                }
                for (int32_t k = 0; k < MTP_ACCEPT_DUMP_TOP_K; ++k) {
                    mtp_accept_dump_write_scalar(target_candidate_ps[(size_t) k]);
                }
                mtp_accept_dump_write_scalar(a.h_scale);
                mtp_accept_dump.write(reinterpret_cast<const char *>(a.h_q8.data()), a.h_q8.size());

                if (!mtp_accept_dump.good()) {
                    SPC_WRN("failed while writing MTP accept/reject dump '%s' after %" PRIu64 " records\n",
                            params.mtp_accept_dump.c_str(), mtp_accept_dump_records);
                    mtp_accept_dump.close();
                    mtp_accept_dump_finished = true;
                    break;
                }

                ++mtp_accept_dump_records;
            }

            finish_mtp_state_dump_if_needed();
            if (is_draft && mtp_state_dump.is_open() && !mtp_state_dump_finished) {
                mtp_state_dump_write_scalar(batch_id);
                mtp_state_dump_write_scalar((int32_t) seq_id);
                mtp_state_dump_write_scalar((int32_t) a.pos);
                mtp_state_dump_write_scalar((int32_t) a.depth);
                mtp_state_dump_write_scalar((int32_t) a.prev_token);
                mtp_state_dump_write_scalar((int32_t) a.draft_token);
                mtp_state_dump_write_scalar(prob);
                mtp_state_dump_write_scalar(accepted);
                mtp_state_dump_write_scalar(verified);
                mtp_state_dump_write_scalar(n_acc);
                mtp_state_dump_write_scalar(n_drafted);
                mtp_state_dump_write_scalar(a.h_in_scale);
                mtp_state_dump_write_scalar(a.h_scale);
                mtp_state_dump.write(reinterpret_cast<const char *>(a.h_in_q8.data()), a.h_in_q8.size());
                mtp_state_dump.write(reinterpret_cast<const char *>(a.h_q8.data()), a.h_q8.size());

                if (!mtp_state_dump.good()) {
                    SPC_WRN("failed while writing MTP transition state dump '%s' after %" PRIu64 " records\n",
                            params.mtp_state_dump.c_str(), mtp_state_dump_records);
                    mtp_state_dump.close();
                    mtp_state_dump_finished = true;
                    break;
                }

                ++mtp_state_dump_records;
            }
            if (is_draft) {
                ++draft_idx;
            }
        }

        attempts.clear();
        finish_mtp_accept_dump_if_needed();
        finish_mtp_state_dump_if_needed();
    }

    void write_mtp_train_record(
            llama_seq_id seq_id,
            llama_pos pos,
            llama_token token,
            const llama_token labels[3],
            const float * h) {
        if (!mtp_dump.is_open() || mtp_dump_finished) {
            return;
        }

        finish_mtp_train_dump_if_needed();
        if (!mtp_dump.is_open()) {
            return;
        }

        float max_abs = 0.0f;
        for (int32_t i = 0; i < n_embd; ++i) {
            max_abs = std::max(max_abs, std::fabs(h[i]));
        }

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        for (int32_t i = 0; i < n_embd; ++i) {
            const float scaled = h[i] / scale;
            const int q = std::max(-127, std::min(127, (int) std::lround(scaled)));
            mtp_dump_q8[i] = (int8_t) q;
        }

        mtp_dump_write_scalar((int32_t) seq_id);
        mtp_dump_write_scalar((int32_t) pos);
        mtp_dump_write_scalar((int32_t) token);
        mtp_dump_write_scalar((int32_t) labels[0]);
        mtp_dump_write_scalar((int32_t) labels[1]);
        mtp_dump_write_scalar((int32_t) labels[2]);
        mtp_dump_write_scalar(scale);
        mtp_dump.write(reinterpret_cast<const char *>(mtp_dump_q8.data()), mtp_dump_q8.size());

        if (!mtp_dump.good()) {
            SPC_WRN("failed while writing MTP training dump '%s' after %" PRIu64 " records\n",
                    params.mtp_train_dump.c_str(), mtp_dump_records);
            mtp_dump.close();
            mtp_dump_finished = true;
            return;
        }

        ++mtp_dump_records;
        finish_mtp_train_dump_if_needed();
    }

    void dump_mtp_train_rows_target(const llama_batch & batch_in, llama_seq_id seq_id, int32_t i_beg, int32_t n_rows) {
        if (!mtp_dump.is_open() || mtp_dump_finished || n_rows < 4) {
            return;
        }

        for (int32_t i = 0; i + 3 < n_rows; ++i) {
            const int32_t k = i_beg + i;
            const llama_token labels[3] = {
                batch_in.token[k + 1],
                batch_in.token[k + 2],
                batch_in.token[k + 3],
            };
            const float * h = verify_h[seq_id].data() + (size_t) i * n_embd;
            write_mtp_train_record(seq_id, batch_in.pos[k], batch_in.token[k], labels, h);
        }
    }

    void dump_mtp_train_rows_draft(const llama_batch & batch_in, llama_seq_id seq_id, int32_t i_beg, int32_t n_rows) {
        if (!mtp_dump.is_open() || mtp_dump_finished || n_rows < 4) {
            return;
        }

        auto * ctx_dft = params.ctx_dft;
        if (ctx_dft == nullptr) {
            return;
        }

        for (int32_t i = 0; i + 3 < n_rows; ++i) {
            const int32_t k = i_beg + i;
            const llama_token labels[3] = {
                batch_in.token[k + 1],
                batch_in.token[k + 2],
                batch_in.token[k + 3],
            };
            const float * h = llama_get_embeddings_nextn_ith(ctx_dft, k);
            write_mtp_train_record(seq_id, batch_in.pos[k], batch_in.token[k], labels, h);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        const float * engram_layer = params.mtp_engram_layer_cache
            ? llama_get_embeddings_layer_inp(ctx_tgt, params.mtp_engram_layer)
            : nullptr;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);

            if (params.mtp_engram_layer_cache) {
                const float * row = engram_layer == nullptr ? nullptr : engram_layer + (size_t) i_batch_end[seq_id] * n_embd;
                set_mtp_engram_current(seq_id, row);
            }

            if (params.mtp_train_dump_source == "draft") {
                dump_mtp_train_rows_draft(batch_in, seq_id, i_batch_beg[seq_id], n_rows);
            } else {
                dump_mtp_train_rows_target(batch_in, seq_id, i_batch_beg[seq_id], n_rows);
            }
        }

        return true;
    }

    void ensure_adaptive_size() {
        const size_t n_pos = (size_t) std::max(1, params.n_max);
        if (accept_pos_ema.size() != n_pos) {
            accept_pos_ema.assign(n_pos, 0.75f);
            cap_scores.assign(n_pos + 1, {});
            accept_updates = 0;
            next_probe_cap = 1;
        }
    }

    int32_t n_max_adaptive() {
        ensure_adaptive_size();

        if (params.n_max <= 1 || accept_updates < MTP_ADAPT_WARMUP) {
            return params.n_max;
        }

        const int32_t n_cap = std::max(1, params.n_max);
        const int32_t min_cap = std::min(n_cap, std::max(1, params.n_min));
        for (int32_t cap = min_cap; cap <= n_cap; ++cap) {
            if (cap_scores[cap].samples < MTP_ADAPT_MIN_CAP_SAMPLES) {
                return cap;
            }
        }

        if (accept_updates % MTP_ADAPT_PROBE_INTERVAL == 0) {
            const int32_t cap = std::max(min_cap, next_probe_cap);
            next_probe_cap = cap % n_cap + 1;
            return cap;
        }

        int32_t best_cap = n_cap;
        double best_score = -1.0;
        for (int32_t cap = min_cap; cap <= n_cap; ++cap) {
            const auto & score = cap_scores[cap];
            if (score.samples == 0) {
                continue;
            }
            if (score.tokens_per_us > best_score) {
                best_score = score.tokens_per_us;
                best_cap = cap;
            }
        }

        return best_cap;
    }

    void update_adaptive_accept(llama_seq_id seq_id, uint16_t n_accepted) {
        ensure_adaptive_size();

        if (params.n_max > 1) {
            ++accept_updates;
            for (size_t pos = 0; pos < accept_pos_ema.size(); ++pos) {
                const float sample = pos < n_accepted ? 1.0f : 0.0f;
                accept_pos_ema[pos] += MTP_ADAPT_ACCEPT_ALPHA * (sample - accept_pos_ema[pos]);
            }
        }

        if (seq_id < 0 || seq_id >= (llama_seq_id) active_cap.size()) {
            return;
        }

        const int32_t cap = active_cap[seq_id];
        active_cap[seq_id] = -1;
        if (cap <= 0 || cap >= (int32_t) cap_scores.size()) {
            return;
        }

        const int64_t elapsed_us = std::max<int64_t>(1, ggml_time_us() - active_start_us[seq_id]);
        const double tokens_per_us = (1.0 + (double) n_accepted) / (double) elapsed_us;
        auto & score = cap_scores[cap];
        if (score.samples == 0) {
            score.tokens_per_us = tokens_per_us;
        } else {
            score.tokens_per_us += MTP_ADAPT_SCORE_ALPHA * (tokens_per_us - score.tokens_per_us);
        }
        ++score.samples;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;
        const int64_t t_start_us = ggml_time_us();

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        const int32_t n_max_eff = n_max_adaptive();
        if (n_max_eff != n_max_adaptive_last) {
            const float p0 = accept_pos_ema.size() > 0 ? accept_pos_ema[0] : 0.0f;
            const float p1 = accept_pos_ema.size() > 1 ? accept_pos_ema[1] : 0.0f;
            const float p2 = accept_pos_ema.size() > 2 ? accept_pos_ema[2] : 0.0f;
            const double s1 = cap_scores.size() > 1 ? cap_scores[1].tokens_per_us * 1e6 : 0.0;
            const double s2 = cap_scores.size() > 2 ? cap_scores[2].tokens_per_us * 1e6 : 0.0;
            const double s3 = cap_scores.size() > 3 ? cap_scores[3].tokens_per_us * 1e6 : 0.0;
            SPC_INF("adaptive draft-mtp cap: n_max_eff=%d configured=%d updates=%u ema=(%.3f, %.3f, %.3f) score_tps=(%.1f, %.1f, %.1f) samples=(%u, %u, %u)\n",
                    n_max_eff, params.n_max, accept_updates, p0, p1, p2, s1, s2, s3,
                    cap_scores.size() > 1 ? cap_scores[1].samples : 0,
                    cap_scores.size() > 2 ? cap_scores[2].samples : 0,
                    cap_scores.size() > 3 ? cap_scores[3].samples : 0);
            n_max_adaptive_last = n_max_eff;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (seq_id >= 0 && seq_id < (llama_seq_id) mtp_accept_pending.size()) {
                mtp_accept_pending[seq_id].clear();
            }
            if (seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_cache_active.size()) {
                mtp_draft_cache_active[seq_id] = {};
            }

            if (!dp.drafting) {
                continue;
            }

            if (try_mtp_draft_cache(seq_id, dp, n_max_eff)) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, pending_h[seq_id].data(), row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
            if (seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_input_h.size()) {
                mtp_draft_input_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            set_mtp_lora_depth(i);
            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                auto & dp = dparams.at(seq_id);

                float p_draft = 0.0f;
                const llama_token id = mtp_fr_select_candidate(dp, cur_p, p_draft);
                maybe_log_mtp_fr_stats();

                // only collect sufficiently confident draft tokens
                if (id == LLAMA_TOKEN_NULL || p_draft < params.p_min) {
                    const llama_token prev_token = dp.result->empty() ? dp.id_last : dp.result->back();
                    const llama_token top_id = cur_p->size > 0 ? cur_p->data[0].id : LLAMA_TOKEN_NULL;
                    const float top_p = cur_p->size > 0 && std::isfinite(cur_p->data[0].p) ? cur_p->data[0].p : p_draft;
                    const float * h_in = seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_input_h.size()
                        ? mtp_draft_input_h[seq_id].data()
                        : nullptr;
                    add_mtp_accept_attempt(seq_id, dp.n_past + (llama_pos) dp.result->size() + 1, prev_token, top_id, i, top_p, h_in, h_row, cur_p, 1);

                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                auto & result = *dp.result;
                const llama_token prev_token = result.empty() ? dp.id_last : result.back();
                const float * h_in = seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_input_h.size()
                    ? mtp_draft_input_h[seq_id].data()
                    : nullptr;
                add_mtp_accept_attempt(seq_id, dp.n_past + (llama_pos) result.size() + 1, prev_token, id, i, p_draft, h_in, h_row, cur_p);

                common_sampler_accept(smpl, id, true);

                result.push_back(id);
                const float * h_next = h_row;
                if (seq_id >= 0 && seq_id < (llama_seq_id) mtp_state_steered_h.size() &&
                        apply_mtp_state_head(prev_token, h_in, h_row, mtp_state_steered_h[seq_id].data())) {
                    h_next = mtp_state_steered_h[seq_id].data();
                }

                if (n_max_eff <= (int) result.size() || (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_next, h_next + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_next, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_next, row_bytes);
                }

                if (seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_input_h.size()) {
                    mtp_draft_input_h[seq_id].assign(h_next, h_next + n_embd);
                }
                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) mtp_accept_pending.size(); ++seq_id) {
            const auto & pending = mtp_accept_pending[seq_id];
            if (pending.empty()) {
                continue;
            }
            bool has_draft = false;
            for (const auto & row : pending) {
                has_draft = has_draft || row.row_type == 0;
            }
            if (!has_draft) {
                dump_mtp_accept_records(seq_id, 0, LLAMA_TOKEN_NULL, nullptr);
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (dp.drafting && !dp.result->empty()) {
                remember_mtp_draft_cache_attempt(seq_id, dp, false);
                const bool from_cache = seq_id >= 0 && seq_id < (llama_seq_id) mtp_draft_cache_active.size() &&
                    mtp_draft_cache_active[seq_id].from_cache;
                if (!from_cache) {
                    active_cap[seq_id] = n_max_eff;
                    active_start_us[seq_id] = t_start_us;
                }
            }
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }
        set_mtp_lora_depth(-1);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, llama_token target_token, bool is_other, const std::vector<common_sampler_accept_trace> * target_trace) override {
        if (!is_other) {
            update_mtp_draft_cache(seq_id, n_accepted);
            update_adaptive_accept(seq_id, n_accepted);
            dump_mtp_accept_records(seq_id, n_accepted, target_token, target_trace);
        }

        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, llama_token /*target_token*/, bool /*is_other*/, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, llama_token /*target_token*/, bool is_other, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, llama_token /*target_token*/, bool is_other, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, llama_token /*target_token*/, bool /*is_other*/, const std::vector<common_sampler_accept_trace> * /*target_trace*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) && params.draft.ctx_dft != nullptr;
        bool has_draft_mtp    = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP))    && params.draft.ctx_dft != nullptr;
        bool has_draft_dflash = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) && params.draft.ctx_dft != nullptr;



        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_draft_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(
        common_speculative * spec,
        llama_seq_id seq_id,
        uint16_t n_accepted,
        llama_token target_token,
        const std::vector<common_sampler_accept_trace> * target_trace) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, target_token, false, target_trace);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, target_token, true, nullptr);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}
