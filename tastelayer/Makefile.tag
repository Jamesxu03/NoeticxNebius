# Tagging targets. Include from the main Makefile with:  include Makefile.tag
PY ?= .venv/bin/python
TAG = $(PY) -m tastelayer.tagging

.PHONY: tag-menu tag-corpus tag-ingredients second-pass consensus tag-report tag-watch

tag-menu:
	$(TAG) tag-dishes --menu-only

tag-corpus:
	$(TAG) tag-dishes

tag-ingredients:
	$(TAG) tag-ingredients

second-pass:
	$(TAG) second-pass

consensus:
	$(TAG) consensus

tag-report:
	$(TAG) report

tag-watch:
	$(TAG) report --watch 600
