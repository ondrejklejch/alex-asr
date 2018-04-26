#!/usr/bin/env bash
set -e

PYTHON=python
FSTDIR=$(python -c "import os,sys; print os.path.realpath(sys.argv[1])" libs/kaldi/tools/openfst)
OPENFST_VERSION=1.3.4
#KALDI_REV=476cb3f6b37c4146057ea5b1f916469fdd2f2273
KALDI_REV=6d6e7a9087c65a88b4bec0b02545b379ee622fc3

if [ -z $1 ]; then
    USE_THREAD=0
else
    USE_THREAD=$1
fi

if [ ! -d libs ]; then
    mkdir -p libs

    # Get Kaldi.
    git clone https://github.com/kaldi-asr/kaldi.git libs/kaldi
    (
        cd libs/kaldi/src;
        git checkout ${KALDI_REV}
    )

    # Get PyFST
    git clone https://github.com/UFAL-DSG/pyfst.git libs/pyfst

    # Prepare Kaldi dependencies.
    (
        # Compile BLAS with multi threaded support
        cd libs/kaldi/tools
        sed -i "s/USE_THREAD=0/USE_THREAD=$USE_THREAD/g" Makefile
        make -j 8 openblas
    )
    (
        # Patch OpenFST makefile so that we can link with it statically.
        cd libs/kaldi/tools;
        sed -i "s/--enable-ngram-fsts/--enable-ngram-fsts --with-pic/g" Makefile
        make -j 8 openfst
    )

    # Configure Kaldi.
    (
        cd libs/kaldi/src;
        git checkout ${KALDI_REV}
        ./configure --shared --use-cuda=no --openblas-root=../tools/OpenBLAS/install
    )

    # Build Kaldi.
    make -j 8 -C libs/kaldi/src
    make -j 8 -C libs/kaldi/src test
else
    echo "It appears that the env is prepared. If there are errors, try deleting libs/ and rerunning the script."
fi
