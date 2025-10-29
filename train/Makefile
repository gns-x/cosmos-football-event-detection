# Makefile for Cosmos VLM Fine-Tuning (VM Side - No Download)
# This file lives in: ~/cosmos_r1_train/train/

# --- 1. VARIABLES ---
SOURCE_VIDEO_DIR = 00_source_videos
CLIP_DIR = 01_clips
ANNO_DIR = 02_annotations
VENV_DIR = cosmos_env

# Auto clipping defaults
AUTO_CLIP_SECONDS ?= 10
AUTO_FPS ?= 4

# List of our 8 event classes
CLASSES = penalty_shot goal goal_line_event woodworks shot_on_target red_card yellow_card hat_trick

# --- 2. RULES ---
.ONESHELL:
.PHONY: step1-setup setup-dirs step1-check clip

# ====================================================================
#  STEP 1A: PROJECT SETUP (MASTER RULE)
#  This rule just creates all the project folders.
# ====================================================================
step1-setup:
	@echo "Creating project directories..."
	@mkdir -p 03_dataset 04_model_output 05_scripts
	@$(foreach class,$(CLASSES),mkdir -p $(CLIP_DIR)/$(class);)
	@$(foreach class,$(CLASSES),mkdir -p $(ANNO_DIR)/$(class);)
	@echo "--------------------------------------------------------"
	@echo "STEP 1 SETUP COMPLETE: All directories created."
	@echo "--------------------------------------------------------"
	@echo "NEXT ACTION: Run 'make step1-check' to verify your videos."
	@echo "--------------------------------------------------------"

# ====================================================================
#  STEP 1B: FILE VERIFICATION
#  Run this *after* 'step1-setup' to check your uploaded files.
# ====================================================================
step1-check:
	@echo "Checking for 8 source video files..."
	@$(foreach class,$(CLASSES), \
		if [ ! -f $(SOURCE_VIDEO_DIR)/$(class)/source.mp4 ]; then \
			echo "ERROR: File not found: $(SOURCE_VIDEO_DIR)/$(class)/source.mp4"; \
			echo "Please make sure your 00_source_videos folder is inside 'train'"; \
			echo "and contains all 8 class folders with 'source.mp4' in each."; \
			exit 1; \
		fi; \
	)
	@echo "--------------------------------------------------------"
	@echo "SUCCESS: All 8 source files found."
	@echo "STEP 1 is 100% complete. Ready for Step 2."
	@echo "--------------------------------------------------------"


# ====================================================================
#  STEP 2: CLIPPING RULE (We will use this in the next step)
# ====================================================================
clip:
	@# If EVENT is not provided, run automatic clipping for all classes
	@if [ -z "$(EVENT)" ]; then \
		echo "Automatic clipping: splitting each class source into $(AUTO_CLIP_SECONDS)s clips @ $(AUTO_FPS) fps"; \
		for class in $(CLASSES); do \
			src="$(SOURCE_VIDEO_DIR)/$$class/source.mp4"; \
			outdir="$(CLIP_DIR)/$$class"; \
			if [ ! -f "$$src" ]; then \
				echo "Skipping $$class: missing $$src"; \
				continue; \
			fi; \
			mkdir -p "$$outdir"; \
			dur_raw=$$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$$src" 2>/dev/null || echo 0); \
			dur_sec=$${dur_raw%%.*}; \
			if [ "$$dur_sec" -le 0 ]; then \
				echo "Skipping $$class: could not read duration (got $$dur_raw)"; \
				continue; \
			fi; \
			clip_len=$(AUTO_CLIP_SECONDS); \
			idx=1; \
			start=0; \
			while [ "$$start" -lt "$$dur_sec" ]; do \
				name=$$(printf "%s_%04d" "$$class" "$$idx"); \
				h=$$((start/3600)); m=$$(((start%3600)/60)); s=$$((start%60)); \
				start_hms=$$(printf "%02d:%02d:%02d" "$$h" "$$m" "$$s"); \
				ffmpeg -hide_banner -loglevel error -y -ss "$$start_hms" -t "$$clip_len" -i "$$src" -vf "fps=$(AUTO_FPS)" -c:v libx264 -preset fast -c:a aac "$$outdir/$$name.mp4" || true; \
				idx=$$((idx+1)); \
				start=$$((start+clip_len)); \
				[ $$start -ge $$dur_sec ] && break; \
			done; \
			echo "Finished $$class -> $$outdir"; \
		done; \
		exit 0; \
	fi
	@# Single manual clip mode when EVENT provided
	@if [ -z "$(START)" ] || [ -z "$(DURATION)" ] || [ -z "$(CLIP_NAME)" ]; then \
		echo "ERROR: Missing required arguments for manual mode."; \
		echo "Usage: make clip EVENT=<class> START=<hh:mm:ss(.ms)> DURATION=<seconds> CLIP_NAME=<name>"; \
		echo "Valid EVENT values: $(CLASSES)"; \
		exit 1; \
	fi
	@if [ ! -f "$(SOURCE_VIDEO_DIR)/$(EVENT)/source.mp4" ]; then \
		echo "ERROR: File not found: $(SOURCE_VIDEO_DIR)/$(EVENT)/source.mp4"; \
		exit 1; \
	fi
	@echo "Clipping $(EVENT) event: $(CLIP_NAME)..."
	@echo "Clipping from file: $(SOURCE_VIDEO_DIR)/$(EVENT)/source.mp4"
	@echo "Start: $(START), Duration: $(DURATION)"
	@mkdir -p "$(CLIP_DIR)/$(EVENT)"
	@ffmpeg -hide_banner -loglevel error -y -ss $(START) -t $(DURATION) -i "$(SOURCE_VIDEO_DIR)/$(EVENT)/source.mp4" -vf "fps=$(AUTO_FPS)" -c:v libx264 -preset fast -c:a aac "$(CLIP_DIR)/$(EVENT)/$(CLIP_NAME).mp4" || { echo "ffmpeg failed"; exit 1; }
	@echo "Saved clip to $(CLIP_DIR)/$(EVENT)/$(CLIP_NAME).mp4"