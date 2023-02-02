from vcr.record_mode import RecordMode
from vcr_langchain import VCR

vcr = VCR(path_transformer=VCR.ensure_suffix(".yaml"), record_mode=RecordMode.NONE)
