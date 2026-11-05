PYTHON ?= python3
FORMATS ?= pdf,png
OUTPUT ?= ../outputs/figures
RUN ?=
CASES ?=
MEDIA_ROOT ?=
ARGS = --formats $(FORMATS) --output $(OUTPUT) $(if $(RUN),--run $(RUN)) $(if $(CASES),--cases $(CASES)) $(if $(MEDIA_ROOT),--media-root $(MEDIA_ROOT))

.PHONY: teaser overview calibration qualitative qualitative-main supplementary gallery localization confidence comparison segmentation
teaser overview calibration qualitative qualitative-main supplementary gallery localization confidence comparison segmentation:
	$(PYTHON) build.py build $@ $(ARGS)
