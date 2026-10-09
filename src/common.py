# -*- coding: utf-8 -*-
"""共享路径与常量：F题 华为杯 2026 求解工程"""
import os

# Resolve all project paths from this file so the pipeline is portable.
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATT = os.path.join(BASE, "real_attachments")
CODE = os.path.join(BASE, "src")
RES = os.path.join(BASE, "results")
FIG = RES

for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)

A = os.path.join(ATT, "A_data_value")
B = os.path.join(ATT, "B_scaling_laws")
C = os.path.join(ATT, "C_efficiency_evolution")

# ---- 17 个配方域与 13 个损失域 ----
MIX_DOMAINS = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
               "dm_mathematics", "github", "philpapers", "stackexchange", "enron_emails",
               "gutenberg_pg_19", "pile_cc", "ubuntu_irc", "europarl", "hackernews",
               "pubmed_abstracts", "uspto_backgrounds"]
LOSS_DOMAINS = ["arxiv", "freelaw", "pubmed_central", "wikipedia_en", "dm_mathematics",
                "github", "stackexchange", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
                "hackernews", "pubmed_abstracts", "uspto_backgrounds"]

def mixture_cols(prefix="train_the_pile"):
    return [f"{prefix}_{d}" for d in MIX_DOMAINS]

def loss_cols():
    return [f"metric/the_pile_{d}_val_loss" for d in LOSS_DOMAINS]
