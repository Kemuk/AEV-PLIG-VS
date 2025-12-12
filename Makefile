# Makefile for Molecular Docking Pipeline
# Convenience wrapper for common Snakemake commands

.PHONY: help setup dry-run run test clean unlock viz

# Default target
help:
	@echo "Molecular Docking Pipeline - Snakemake Workflow"
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Set up conda environments"
	@echo "  dry-run     - Test workflow without executing"
	@echo "  run         - Run complete pipeline"
	@echo "  test        - Run small test on first protein"
	@echo "  clean       - Remove intermediate files"
	@echo "  clean-all   - Remove all generated files"
	@echo "  unlock      - Unlock workflow if stuck"
	@echo "  viz         - Generate workflow visualization"
	@echo "  report      - Generate HTML report"
	@echo ""
	@echo "Options:"
	@echo "  CORES=N     - Number of cores (default: 8)"
	@echo "  PROFILE=P   - Snakemake profile (default: none)"
	@echo ""
	@echo "Examples:"
	@echo "  make dry-run"
	@echo "  make run CORES=32"
	@echo "  make run PROFILE=slurm"

# Configuration
CORES ?= 8
PROFILE ?=

# Snakemake command base
SNAKE = snakemake --cores $(CORES) --use-conda
ifdef PROFILE
  SNAKE += --profile $(PROFILE)
endif

# Setup conda environments
setup:
	@echo "Creating conda environments..."
	conda env create -f envs/vscreen.yaml
	conda env create -f envs/aev_plig.yaml
	@echo "Done! Environments created."

# Dry run
dry-run:
	@echo "Running dry-run (no execution)..."
	$(SNAKE) --dry-run --printshellcmds

# Run complete pipeline
run:
	@echo "Running complete pipeline with $(CORES) cores..."
	$(SNAKE)

# Test on small subset
test:
	@echo "Testing on first protein only..."
	$(SNAKE) --until mol2_to_pdbqt --cores 1

# Clean intermediate files
clean:
	@echo "Cleaning intermediate files..."
	$(SNAKE) clean
	@echo "Done!"

# Clean all generated files
clean-all:
	@echo "WARNING: This will remove ALL generated files!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(SNAKE) clean_all; \
		echo "Done!"; \
	fi

# Unlock workflow
unlock:
	@echo "Unlocking workflow..."
	snakemake --unlock
	@echo "Done!"

# Visualize workflow
viz:
	@echo "Generating workflow visualization..."
	snakemake --dag | dot -Tpng > workflow_dag.png
	snakemake --rulegraph | dot -Tpng > workflow_rules.png
	@echo "Created: workflow_dag.png and workflow_rules.png"

# Generate HTML report
report:
	@echo "Generating workflow report..."
	$(SNAKE) --report report.html
	@echo "Created: report.html"

# List all rules
list-rules:
	@snakemake --list

# Show file status
status:
	@snakemake --summary

# Detailed file status
detailed-status:
	@snakemake --detailed-summary

# Archive workflow
archive:
	@echo "Creating workflow archive..."
	tar -czf workflow_$(shell date +%Y%m%d).tar.gz \
		Snakefile config.yaml envs/ scripts/ README.md MIGRATION_GUIDE.md
	@echo "Created: workflow_$(shell date +%Y%m%d).tar.gz"
