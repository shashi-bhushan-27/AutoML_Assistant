import sys

# Windows only: scikit-learn 1.3's bundled OpenMP runtime (sklearn/.libs/vcomp140.dll) makes torch's
# c10.dll fail to initialise (WinError 1114) if scikit-learn is imported first. torch is needed for the
# retrieval embeddings (sentence-transformers), so load it before anything imports scikit-learn.
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pass
