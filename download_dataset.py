#!/usr/bin/env python3
"""Temporary script to download the MCP-Atlas dataset."""

from datasets import load_dataset
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

print("Downloading ScaleAI/MCP-Atlas dataset...")
ds = load_dataset("ScaleAI/MCP-Atlas")

print("Saving dataset to data folder...")
ds.save_to_disk("data/MCP-Atlas")

print("Done! Dataset saved to data/MCP-Atlas")
