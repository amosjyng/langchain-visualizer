from vcr_langchain import VCR

vcr = VCR(path_transformer=VCR.ensure_suffix(".yaml"))
