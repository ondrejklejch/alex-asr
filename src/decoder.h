#ifndef ALEX_ASR_DECODER_
#define ALEX_ASR_DECODER_
#include <vector>
#include <memory>
#include "fst/fst-decl.h"
#include "base/kaldi-types.h"

#include "src/decoder_config.h"
#include "src/feature_pipeline.h"

#include "feat/online-feature.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice.h"
#include "nnet2/online-nnet2-decodable.h"
#include "online2/online-gmm-decodable.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-endpoint.h"


using namespace kaldi;

namespace alex_asr {
    class Decoder {
    public:
        Decoder(const string model_path);
        ~Decoder();

        int32 Decode(int32 max_frames);
        void FrameIn(unsigned char *buffer, int32 buffer_length);
        void FrameIn(VectorBase<BaseFloat> *waveform_in);
        bool GetBestPath(std::vector<int> *v_out, BaseFloat *prob);
        bool GetLattice(fst::VectorFst<fst::LogArc> * out_fst, double *tot_lik, bool end_of_utt=true);
        bool GetTimeAlignment(std::vector<int> *words, std::vector<int> *times, std::vector<int> *lengths);
        bool GetTimeAlignmentWithWordConfidence(std::vector<int> *words, std::vector<int> *times, std::vector<int> *lengths, std::vector<float> *confs);
        string GetWord(int word_id);
        void InputFinished();
        bool EndpointDetected();
        void FinalizeDecoding();
        void Reset();
        float FinalRelativeCost();
        int32 NumFramesDecoded();
        int32 TrailingSilenceLength();
        void GetIvector(std::vector<float> *ivector);
        void SetBitsPerSample(int n_bits);
        int GetBitsPerSample();
        float GetFrameShift();
    private:
        FeaturePipeline *feature_pipeline_;

        fst::StdFst *hclg_;
        LatticeFasterOnlineDecoder *decoder_;
        TransitionModel *trans_model_;
        nnet2::AmNnet *am_nnet2_;
        nnet3::AmNnetSimple *am_nnet3_;
        AmDiagGmm *am_gmm_;
        fst::SymbolTable *words_;
        WordBoundaryInfo *word_boundary_info_;
        DecoderConfig *config_;
        DecodableInterface *decodable_;

        void InitTransformMatrices();
        void LoadDecoder();
        void ParseConfig();
        void Deallocate();
        bool FileExists(const std::string& name);
    };

/// @} end of "addtogroup online_latgen"

} // namespace kaldi

#endif  // #ifdef PYKALDI2_DECODER_H_
