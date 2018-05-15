#include "src/decoder.h"
#include "src/utils.h"

#include "online2/onlinebin-util.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"
#include "lat/sausages.h"
#include "nnet3/nnet-utils.h"

using namespace kaldi;

namespace alex_asr {
    Decoder::Decoder(const string model_path) :
            feature_pipeline_(NULL),
            hclg_(NULL),
            lm_small_(NULL),
            lm_big_(NULL),
            decoder_(NULL),
            trans_model_(NULL),
            am_nnet2_(NULL),
            am_nnet3_(NULL),
            nnet3_info_(NULL),
            am_gmm_(NULL),
            words_(NULL),
            config_(NULL),
            decodable_(NULL)
    {
        // Change dir to model_path. Change back when leaving the scope.
        local_cwd cwd_to_model_path(model_path);
        KALDI_VLOG(2) << "Decoder is setting up models: " << model_path;

        ParseConfig();
        LoadDecoder();
        Reset();

        KALDI_VLOG(2) << "Decoder is successfully initialized.";
    }

    Decoder::~Decoder() {
        delete feature_pipeline_;
        delete hclg_;
        delete lm_small_;
        delete lm_big_;
        delete decoder_;
        delete trans_model_;
        delete am_nnet2_;
        delete am_nnet3_;
        delete nnet3_info_;
        delete am_gmm_;
        delete words_;
        delete config_;
        delete decodable_;
    }

    void Decoder::ParseConfig() {
        KALDI_PARANOID_ASSERT(config_ == NULL);

        config_ = new DecoderConfig();

        string cfg_name;
        if(FileExists("pykaldi.cfg")) {
            cfg_name = "pykaldi.cfg";
            KALDI_WARN << "Using deprecated configuration file. Please move pykaldi.cfg to alex_asr.conf.";
        } else if(FileExists("alex_asr.conf")) {
            cfg_name = "alex_asr.conf";
        } else {
            KALDI_ERR << "AlexASR Decoder configuration (alex_asr.conf) not found in model directory."
                    "Please check your configuration.";
        }

        config_->LoadConfigs(cfg_name);

        if(!config_->InitAndCheck()) {
            KALDI_ERR << "Error when checking if the configuration is valid. "
                    "Please check your configuration.";
        }
    }

    bool Decoder::FileExists(const std::string& name) {
        struct stat buffer;
        return (stat (name.c_str(), &buffer) == 0);
    }

    void Decoder::LoadDecoder() {
        bool binary;
        Input ki(config_->model_rxfilename, &binary);

        KALDI_PARANOID_ASSERT(trans_model_ == NULL);
        trans_model_ = new TransitionModel();
        trans_model_->Read(ki.Stream(), binary);

        if(config_->model_type == DecoderConfig::GMM) {
            KALDI_PARANOID_ASSERT(am_gmm_ == NULL);
            am_gmm_ = new AmDiagGmm();
            am_gmm_->Read(ki.Stream(), binary);
        } else if(config_->model_type == DecoderConfig::NNET2) {
            KALDI_PARANOID_ASSERT(am_nnet2_ == NULL);
            am_nnet2_ = new nnet2::AmNnet();
            am_nnet2_->Read(ki.Stream(), binary);
        } else if(config_->model_type == DecoderConfig::NNET3) {
            KALDI_PARANOID_ASSERT(am_nnet3_ == NULL);
            am_nnet3_ = new nnet3::AmNnetSimple();
            am_nnet3_->Read(ki.Stream(), binary);
            SetBatchnormTestMode(true, &am_nnet3_->GetNnet());
            SetDropoutTestMode(true, &am_nnet3_->GetNnet());
            CollapseModel(nnet3::CollapseModelConfig(), &am_nnet3_->GetNnet());
            nnet3_info_ = new nnet3::DecodableNnetSimpleLoopedInfo(config_->nnet3_decodable_opts, am_nnet3_);
        }

        KALDI_PARANOID_ASSERT(hclg_ == NULL);
        hclg_ = ReadDecodeGraph(config_->fst_rxfilename);
        KALDI_PARANOID_ASSERT(decoder_ == NULL);
        decoder_ = new LatticeFasterOnlineDecoder(*hclg_, config_->decoder_opts);

        KALDI_PARANOID_ASSERT(words_ == NULL);
        words_ = fst::SymbolTable::ReadText(config_->words_rxfilename);

        KALDI_PARANOID_ASSERT(word_boundary_info == NULL);
        if(config_->word_boundary_rxfilename != "") {
            WordBoundaryInfoNewOpts word_boundary_info_opts;
            word_boundary_info_ = new WordBoundaryInfo(word_boundary_info_opts, config_->word_boundary_rxfilename);
        }

        if(config_->rescore == true) {
            LoadLM(config_->lm_small_rxfilename, &lm_small_);
            LoadLM(config_->lm_big_rxfilename, &lm_big_);
        }
    }

    void Decoder::LoadLM(
        const string path,
        fst::MapFst<fst::StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> > **lm_fst
    ) {
        int num_states_cache = 50000;

        if (FileExists(path)) {
            fst::VectorFst<fst::StdArc> *std_lm_fst = fst::ReadFstKaldi(path);
            fst::Project(std_lm_fst, fst::PROJECT_OUTPUT);
            if (std_lm_fst->Properties(fst::kILabelSorted, true) == 0) {
                fst::ILabelCompare<fst::StdArc> ilabel_comp;
                fst::ArcSort(std_lm_fst, ilabel_comp);
            }

            fst::CacheOptions cache_opts(true, num_states_cache);
            fst::MapFstOptions mapfst_opts(cache_opts);
            fst::StdToLatticeMapper<BaseFloat> mapper;
            *lm_fst = new fst::MapFst<fst::StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> >(*std_lm_fst, mapper, mapfst_opts);
            delete std_lm_fst;

            KALDI_VLOG(2) << "LM loaded: " << path;
        } else {
            KALDI_ERR << "LM" << path << "doesn't exist.";
        }
    }

    void Decoder::Reset() {
        delete feature_pipeline_;
        delete decodable_;

        feature_pipeline_ = new FeaturePipeline(*config_);

        if(config_->model_type == DecoderConfig::GMM) {
            decodable_ = new DecodableDiagGmmScaledOnline(*am_gmm_,
                                                          *trans_model_,
                                                          config_->decodable_opts.acoustic_scale,
                                                          feature_pipeline_->GetFeature());
        } else if(config_->model_type == DecoderConfig::NNET2) {
            decodable_ = new nnet2::DecodableNnet2Online(*am_nnet2_,
                                                         *trans_model_,
                                                         config_->decodable_opts,
                                                         feature_pipeline_->GetFeature());
        } else if(config_->model_type == DecoderConfig::NNET3) {
            decodable_ = new nnet3::DecodableAmNnetLoopedOnline(*trans_model_,
                                                                *nnet3_info_,
                                                                feature_pipeline_->GetInputFeature(),
                                                                feature_pipeline_->GetIvectorFeature());
        } else {
            KALDI_ASSERT(false);  // This means the program is in invalid state.
        }

        decoder_->InitDecoding();
    }

    bool Decoder::EndpointDetected() {
        return kaldi::EndpointDetected(config_->endpoint_config, *trans_model_,
                                       config_->FrameShiftInSeconds(),
                                       *decoder_);
    }

    void Decoder::FrameIn(VectorBase<BaseFloat> *waveform_in) {
        feature_pipeline_->AcceptWaveform(config_->SamplingFrequency(), *waveform_in);
    }

    void Decoder::FrameIn(unsigned char *buffer, int32 buffer_length) {
        int n_frames = buffer_length / (config_->bits_per_sample / 8);

        Vector<BaseFloat> waveform(n_frames);

        for(int32 i = 0; i < n_frames; ++i) {
            switch(config_->bits_per_sample) {
                case 8:
                {
                    waveform(i) = (*buffer);
                    buffer++;
                    break;
                }
                case 16:
                {
                    int16 k = *reinterpret_cast<uint16*>(buffer);
#ifdef __BIG_ENDDIAN__
                    KALDI_SWAP2(k);
#endif
                    waveform(i) = k;
                    buffer += 2;
                    break;
                }
                default:
                    KALDI_ERR << "Unsupported bits ber sample (implement yourself): "
                    << config_->bits_per_sample;
            }
        }
        this->FrameIn(&waveform);
    }

    void Decoder::InputFinished() {
        feature_pipeline_->InputFinished();
    }

    int32 Decoder::Decode(int32 max_frames) {
        int32 decoded = decoder_->NumFramesDecoded();
        decoder_->AdvanceDecoding(decodable_, max_frames);

        return decoder_->NumFramesDecoded() - decoded;
    }

    void Decoder::FinalizeDecoding() {
        decoder_->FinalizeDecoding();
    }

    bool Decoder::GetBestPath(std::vector<int> *out_words, BaseFloat *prob) {
        *prob = -1.0f;

        Lattice lat;

        bool ok = decoder_->GetBestPath(&lat);

        // TODO: BEST PATH can't work  with reweighting of AM because it returns a single path before the rew.

        LatticeWeight weight;
        std::vector<int32> ids;
        fst::GetLinearSymbolSequence(lat,
                                     static_cast<std::vector<int32> *>(0),
                                     out_words,
                                     &weight);

        *prob = weight.Value1() + weight.Value2();

        return ok;
    }

    bool Decoder::GetPrunedLattice(CompactLattice *lat) {
        Lattice raw_lat;

        if (decoder_->NumFramesDecoded() == 0)
            KALDI_ERR << "You cannot get a lattice if you decoded no frames.";

        if (!config_->decoder_opts.determinize_lattice)
            KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";


        bool ok = decoder_->GetRawLattice(&raw_lat);
        if (config_->model_type == DecoderConfig::NNET3) {
            if (config_->post_decode_acwt != 1.0) {
                PostDecodeAMRescore(&raw_lat, config_->post_decode_acwt);
            }
        }

        BaseFloat lat_beam = config_->decoder_opts.lattice_beam;
        if(!config_->rescore) {
            DeterminizeLatticePhonePrunedWrapper(*trans_model_, &raw_lat, lat_beam, lat, config_->decoder_opts.det_opts);
        } else {
            CompactLattice pruned_lat;

            DeterminizeLatticePhonePrunedWrapper(*trans_model_, &raw_lat, lat_beam, &pruned_lat, config_->decoder_opts.det_opts);
            ok = ok && RescoreLattice(pruned_lat, lat);
        }

        return ok;
    }

    bool Decoder::RescoreLattice(CompactLattice lat, CompactLattice *rescored_lattice) {
        CompactLattice intermidiate_lattice;
        bool ok = true;

        ok = ok && RescoreLatticeWithLM(lat, -1.0, lm_small_, &intermidiate_lattice);
        ok = ok && RescoreLatticeWithLM(intermidiate_lattice, 1.0, lm_big_, rescored_lattice);

        return ok;
    }

    bool Decoder::RescoreLatticeWithLM(
        CompactLattice lat,
        float lm_scale,
        fst::MapFst<fst::StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> > *lm_fst,
        CompactLattice *rescored_lattice) {

        // Taken from https://github.com/kaldi-asr/kaldi/blob/9b9b561e2b3d2bdf64c84f8626175953f0885264/src/latbin/lattice-lmrescore.cc

        Lattice lattice;
        ConvertLattice(lat, &lattice);

        fst::ScaleLattice(fst::GraphLatticeScale(1.0 / lm_scale), &lattice);
        ArcSort(&lattice, fst::OLabelCompare<LatticeArc>());

        Lattice composed_lat;
        fst::TableComposeOptions compose_opts(fst::TableMatcherOptions(), true, fst::SEQUENCE_FILTER, fst::MATCH_INPUT);
        fst::TableComposeCache<fst::Fst<LatticeArc> > lm_compose_cache(compose_opts);
        TableCompose(lattice, *lm_fst, &composed_lat, &lm_compose_cache);
        Invert(&composed_lat);

        DeterminizeLattice(composed_lat, rescored_lattice);
        fst::ScaleLattice(fst::GraphLatticeScale(lm_scale), rescored_lattice);

        return rescored_lattice->Start() != fst::kNoStateId;
    }

    void Decoder::PostDecodeAMRescore(Lattice *lat, double acoustic_scale) {
        std::vector<std::vector<double> > scale(2);
        scale[0].resize(2);
        scale[1].resize(2);
        scale[0][0] = 1.0; // lm scale
        scale[0][1] = 0.0; // acoustic cost to lm cost scaling
        scale[1][0] = 0.0; // lm cost to acoustic cost scaling
        scale[1][1] = acoustic_scale;

        fst::ScaleLattice(scale, lat);
    }

    bool Decoder::GetLattice(fst::VectorFst<fst::LogArc> *fst_out,
                                     double *tot_lik, bool end_of_utterance) {
        CompactLattice lat;
        bool ok = true;

        ok = this->GetPrunedLattice(&lat);
        *tot_lik = CompactLatticeToWordsPost(lat, fst_out);

        return ok;
    }

    bool Decoder::GetTimeAlignment(std::vector<int> *words, std::vector<int> *times, std::vector<int> *lengths) {
        CompactLattice compact_lat;
        CompactLattice best_path;
        CompactLattice aligned_best_path;
        bool ok = true;

        ok = this->GetPrunedLattice(&compact_lat);
        CompactLatticeShortestPath(compact_lat, &best_path);

        if(config_->word_boundary_rxfilename == "") {
            ok = ok && CompactLatticeToWordAlignment(best_path, words, times, lengths);
        } else {
            ok = ok && WordAlignLattice(best_path, *trans_model_, *word_boundary_info_, 0, &aligned_best_path);
            ok = ok && CompactLatticeToWordAlignment(aligned_best_path, words, times, lengths);
        }

        return ok;
    }

    bool Decoder::GetTimeAlignmentWithWordConfidence(std::vector<int> *words, std::vector<int> *times, std::vector<int> *lengths, std::vector<float> *confs) {
        Lattice lat;
        CompactLattice compact_lat;
        CompactLattice best_path;
        CompactLattice aligned_best_path;
        bool ok = true;

        ok = this->GetPrunedLattice(&compact_lat);
        CompactLatticeShortestPath(compact_lat, &best_path);

        if(config_->word_boundary_rxfilename != "") {
            ok = ok && WordAlignLattice(best_path, *trans_model_, *word_boundary_info_, 0, &aligned_best_path);
        } else {
            aligned_best_path = best_path;
        }

        ok = ok && CompactLatticeToWordAlignment(aligned_best_path, words, times, lengths);
        MinimumBayesRisk *mbr = new MinimumBayesRisk(compact_lat, *words, MinimumBayesRiskOptions());
        *confs = mbr->GetOneBestConfidences();

        return ok;
    }

    string Decoder::GetWord(int word_id) {
        return words_->Find(word_id);
    }

    float Decoder::FinalRelativeCost() {
        return decoder_->FinalRelativeCost();
    }

    int32 Decoder::NumFramesDecoded() {
        return decoder_->NumFramesDecoded();
    }

    int32 Decoder::TrailingSilenceLength() {
        if(config_->endpoint_config.silence_phones == "") {
            KALDI_WARN << "Trying to get training silence length for a model that does not have"
                          "silence phones configured.";
            return -1;
        } else {
            return kaldi::TrailingSilenceLength(*trans_model_,
                                                config_->endpoint_config.silence_phones,
                                                *decoder_);
        }
    }

    void Decoder::GetIvector(std::vector<float> *ivector) {
        if(config_->use_ivectors) {
            KALDI_WARN << "Trying to get an Ivector for a model that does not have Ivectors.";
        } else {
            OnlineIvectorFeature *ivector_ftr = feature_pipeline_->GetIvectorFeature();

            Vector<BaseFloat> ivector_res;
            ivector_res.Resize(ivector_ftr->Dim());
            ivector_ftr->GetFrame(decoder_->NumFramesDecoded() - 1, &ivector_res);

            BaseFloat *data = ivector_res.Data();
            for (int32 i = 0; i < ivector_res.Dim(); i++) {
                ivector->push_back(data[i]);
            }
        }
    }

    void Decoder::SetBitsPerSample(int n_bits) {
        KALDI_ASSERT(n_bits % 8 == 0);

        config_->bits_per_sample = n_bits;
    }

    int Decoder::GetBitsPerSample() {
        return config_->bits_per_sample;
    }

    float Decoder::GetFrameShift() {
        return config_->FrameShiftInSeconds();
    }

    float Decoder::GetSamplingFrequency() {
        return (float) config_->SamplingFrequency();
    }
}
