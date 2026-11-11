# Facts sheet for the JCTC revision (ct-2026-01838j)

Everything a section writer needs that is NOT already in `paper/body.tex` (the
submitted text), `paper/macros.tex` (derived numbers) or `results/*.md`. Read
`reviewercommentsOct2026` first. Nothing below may be contradicted; if a writer
needs a fact not listed here or in the repo, write `[[NEEDS FACT: ...]]` rather
than inventing it.

## 1. What the reviewers asked for, and the paper's answer

| # | Reviewer point | Answer in the revision |
|---|---|---|
| R1.1, R2.2 | Terminology (Factor A/B, oracle, preservation, lane, gate) impenetrable; pronoun-heavy; methods in broad strokes | Full plain-language rewrite, conventional structure, notation table, no bespoke terms (vocabulary map in §3 below) |
| R1.2 | Committees not characterised (training, diversity, loss, uncertainty definition, calibration) | New Methods §2.1-2.2 with the complete recipes in §4 below; new calibration metrics (z-score variance, reliability curves) |
| R1.3 | No energy/force MAE/RMSE; no error-vs-uncertainty plot | New Table 1 (`results/records/j12_accuracy.json`, all systems) and a new calibration figure (per-atom error vs committee std, reliability curves) |
| R1.4, R2 "Table 1" | MPtraj result: MACE-MP trunk trained on MPtraj; near-equilibrium zero-force crystals; force-norm conclusion too broad; error identification is not acquisition | (i) acknowledge trunk pretraining overlap explicitly; (ii) J10: out-of-distribution scan on OMat24 (which post-dates MACE-MP-0 and is what Beck et al. used as OOD test) with all four foundation committees; (iii) filtered MPtraj re-analysis excluding max‖F_ref‖ < 0.05 eV/Å and single-atom cells, plus AUROC within reference-force deciles; (iv) J11: one real active-learning round on 3BPA with retraining; (v) claims rescoped to what these show |
| R2.1 | "comparison to true error is uncommon" is false; "size dependence" unsupported | Retract the first (the UQ literature routinely compares to true error: Kahle & Zipoli 2022, Tan et al. 2023, Busk et al. 2021, Kellner & Ceriotti 2024, Beck et al. 2025); reframe: the *sketching/approximation* literature validates against the exact committee, and we evaluate both links. Support the second with Smith et al. 2018 (√N normalisation), Kellner & Ceriotti 2024 (σ_V ∝ N vs √N), Lortaraprasert et al. 2026 (arXiv:2610.40126) |
| R2.3 | No field introduction; cite 10.1088/2632-2153/ad594a, 10.1088/2632-2153/ae0bf0, 10.1515/revce-2024-0028 and the papers they cover | New introduction (~1.5 pages) citing the keys in §6 below |
| R2 detail | Fig 1/2 mix "system + committee type" and "system" | All six labelled "system, construction": 3BPA disjoint heads / overlapping heads / identical data; ethanol, aspirin, azobenzene all "disjoint heads" (one jointly trained 10-molecule rMD17 committee) |
| R2 detail | force-norm "revolutionary result" | It is not claimed as revolutionary: force error grows with force magnitude (a known effect), so the predicted force norm is a strong *baseline*. The claim is that committee methods must be scored against it, and the OOD + filtered + retraining evidence bounds where it holds |
| defect | body.tex:288 pairs water-MD naive 2.28× with 3BPA-same shared 25.77× as "matched" | Report the matched pairs: 3BPA identical data 2.58× vs 25.77×; 3BPA resampled 2.56× vs 13.52×; water MD 2.28× vs 6.40×; water PIMD 3.08× vs 24.08× |
| defect | body.tex:313 cites batatia2022mace for MACE-MP | cite batatia2023foundation for MACE-MP, batatia2022mace for the MACE architecture |

## 2. Section plan (file names under paper/sections/)

01_introduction.tex — Introduction (incl. what was previously "Related work")
02_methods.tex — Methods: 2.1 committees; 2.2 uncertainty definitions and the three structure-level scores; 2.3 zero-cost baselines; 2.4 error targets and ranking metric; 2.5 calibration metrics; 2.6 cheap screening rule and its conformal constant; 2.7 datasets, splits and contamination statement; 2.8 out-of-distribution and active-learning protocols; notation table
03_results.tex — Results (written in pass 2, when J10/J11/J12 numbers exist): 3.1 accuracy and calibration; 3.2 disagreement ranks error at fixed size; 3.3 the screening rule and why randomised sketches fail; 3.4 committee size and independently trained committees; 3.5 condensed phase and distribution shift; 3.6 materials pool: in-distribution, filtered, out-of-distribution; extensivity; the zero-cost baseline; 3.7 one active-learning round; 3.8 wall clock
04_discussion.tex — Discussion, recommendations, limitations
05_availability.tex — Data and Software Availability; Use of Large Language Models

## 3. Vocabulary map (old term → what to write instead; old terms are BANNED everywhere, including captions)

- "Factor A" / "oracle quality" → "error-ranking quality": whether the committee's disagreement identifies the structures on which the model's forces are most wrong
- "Factor B" / "preservation" / "decision agreement" → "fidelity of the cheap screening rule to the exact committee statistic"
- "oracle" → "exact committee statistic" (or "the exact committee")
- "lane" → "backward pass" (one head-space projection g_k = F w_k costs one reverse-mode pass through the trunk); "lane budget" → "number of backward passes"
- "gate" / "calibrated gate" → "screening rule" (a rule that skips the exact computation when a cheap score times a calibration constant falls below a threshold)
- "free signal" → "zero-cost baseline" (two of them: the predicted force norm, max over atoms of ‖mean force‖, which is a by-product of the forward pass; and the spread of the head energies, which costs one forward pass and no backward pass)
- "global statistic" → keep, but define at first use as "the sum over all 3N coordinates of the per-coordinate head variance"; "max-component" and "max-atom" likewise defined at first use
- "leading-only estimator" → "deterministic leading-subspace estimator"; "control variate" → "randomised low-rank sketch with control variate" (define once)
- "naive committee" → "independently trained committee" (Beck et al. call it naive; say so once)
- "shared-trunk" / "multi-head committee" → keep both, define once: M readout heads on one message-passing trunk
- Never use "the former/the latter", "others", or a pronoun whose referent is not in the same sentence.

## 4. How every committee was built (from the Zenodo archive's run_mace.sh / config.yaml; cite beck2025committee)

Common MACE architecture for the custom committees: 2 interaction layers, 32 channels, max L = 1, correlation order 3, r_max = 6.0 Å, loss = 10·MSE(energy) + 1000·MSE(forces) (per-atom-normalised as in MACE), batch 10, Adam + EMA, stochastic weight averaging over the final 25% of epochs, float32, seed 123. Heads: each committee member is a separate readout attached to every layer's features (linear readouts on the first layer, a 16-hidden-node SiLU MLP on the last), all heads sharing the trunk. Head diversity comes from the training-data assignment:
- disjoint (`--disjoint_committee`): the training set is split evenly across the 8 heads with no overlap; each structure updates the trunk and its own head only.
- overlapping (`--valid_fraction=0.2`, no fixed validation file): each head sees an independently drawn random 80% of the training set.
- "same" / identical data: all heads see the full training set; heads differ only by their random initialisation.
Epochs (SWA start): 3BPA disjoint 500 (375), overlapping 78 (59), same 65 (49); water disjoint 200 (150), overlapping 31 (23); rMD17 disjoint 320 (240).
Independently trained ("naive") committees: 8 full MACE models of the same architecture, seeds 0-7; `naive-committee-same` uses identical train/validation files; `naive-committee` uses `--valid_fraction=0.2` per seed. Epochs: 3BPA naive 625 (469), naive-same 521 (391); water naive 250 (188).
Training sets: 3BPA 100 frames at 300 K (`train_300K_100.xyz`), validation 300 frames at 300 K; tests at 300/600/1200 K (1669/2138/2139 frames; 27 atoms; from Kovács et al. 2021). rMD17: one joint committee trained on 50 frames of each of the 10 rMD17 molecules (500 frames; validation 100 per molecule), tested on 1000 held-out frames per molecule for ethanol (9 atoms), aspirin (21), azobenzene (24). Water: 111 training frames of 64 molecules (192 atoms, periodic, 12.42 Å cube), 500 validation; tests 500 classical-MD and 500 PIMD frames at 300 K.
Foundation committees: MACE-MP-0b "medium" (agnesi radial basis) trunk, frozen (`optimize_readouts_only: true`), 8 new disjoint readout heads (~15k parameters), fine-tuned 1000 epochs (SWA from 800), lr 0.01, AMSGrad, EMA 0.99, float64, loss 10·E + 1000·F + 1·stress, E0s from the foundation model. Four variants differ only in the 8,000 (or 1,000) MPtraj structures the heads were fitted on: QbC-selected 8,000 (the one scored on MPtraj in J6B, because the pool is its complement), QbC-selected 1,000, random 8,000, max-mean-force 8,000.
The MACE-MP-0b trunk was pretrained on MPtraj, so the MPtraj pool (136,923 structures, 89 elements, 1-444 atoms, median 20 atoms) is in-distribution for the trunk and out-of-sample only for the readouts. This was not acknowledged in the submitted version and must be stated plainly.

## 5. Uncertainty and error definitions (match experiments/j2a_oracle_panel.py)

F ∈ R^{3N×M}: per-head forces. v_d = Var over heads of F_d (unbiased, M−1). Per-atom std σ_a = mean over xyz of sqrt(v_d) (Beck et al. Eq. 4). Structure-level scores: global = Σ_d v_d; max-component = max_d sqrt(v_d); max-atom = max_a σ_a. Zero-cost baselines: force norm = max_a ‖mean_m F_a‖; energy spread = std over heads of total energy. Error targets: Δ = mean_m F_m − F_DFT; e_max = max_a ‖Δ_a‖ (primary); e_rmse; e_maxcomp = max_d |Δ_d|; e_q95. Positives = top 5% of structures by the error target; AUROC = Mann–Whitney. Overconfidence ratio = force RMSE of the committee mean ÷ median per-atom σ_a. z-score variance = mean of (Δ_d/σ_d)^2 over components (1 for a calibrated Gaussian; the shared-trunk committees give 5–430, independent ones ≈ 3.6; see results/records/j12_accuracy.json). Spearman ρ between per-atom σ_a and per-atom ‖Δ_a‖ is 0.56–0.78 on molecules, 0.30–0.64 on water.

## 6. Citation keys available in paper/references.bib (do not invent keys)

Existing: beck2025committee, batatia2022mace, kovacs2021beyond, christensen2020role, deng2023chgnet, batatia2023foundation, seung1992query, vovk2005algorithmic, lei2018distribution, meyer2021hutchpp, kunsch1989jackknife.
Added: kahle2022quality, tan2023single, kellner2024shallow, vinchurkar2025gnn, dai2025review, carrete2023deep, imbalzano2021uncertainty, schran2020committee, zhang2019active, smith2018less, podryabinkin2017active, vandermause2020flare, zhu2023fast, busk2021calibrated, pernot2022long, hu2022conformal, bilbrey2025foundation, bigi2024rigidity, heid2024spatially, kulichenko2023uncertainty, schwalbekoda2025information, janet2019quantitative, musil2019fast, riebesell2025matbench, schafer2026shallow, levi2022evaluating, tran2020methods, artrith2012high, gastegger2017machine, zaverkin2021exploration, barrosoluque2024omat24, lortaraprasert2026size, chepkasov2026active, wollschlager2023uncertainty, lakshminarayanan2017simple, krogh1995neural.

## 7. New experiments (pre-registered; results arrive in pass 2)

- J10 (protocols/j10_ood.yaml): OMat24 validation subsets aimd-from-PBE-1000-nvt, aimd-from-PBE-1000-npt, rattled-1000-subsampled; 10,000 structures each, seed 20261111; all four foundation committees; predictions P1-P5.
- J11 (protocols/j11_active_learning.yaml): 3BPA, pool = 1200 K test set (2,139 frames), k = 50 per rule, rules none/global/maxcomp/maxatom/force_norm/energy_std/random, seeds 0-2, single-head MACE with the independent-committee recipe, evaluated on the 1,820 pool frames no rule selected, plus the 600 K and 300 K test sets; predictions P1-P5.
- J12 (experiments/j12_accuracy.py): accuracy and calibration table for every system (done for molecules and water; materials rows follow the scans).

## 8. Style for this venue

JCTC article, achemso class, natbib (\citep, \citet). British or American spelling consistently (the submitted text is British: "labelled", "normalisation"; keep British). No em-dashes in new prose; no "we argue" more than once per section; every number in prose comes from a \fs... macro, never typed. Figures/tables referenced by \ref with the existing labels: fig:factora, fig:gate, fig:beta, fig:extensivity, fig:speedup, tab:mptraj, tab:pairs; new ones: tab:accuracy, fig:calibration, tab:ood, tab:al.
