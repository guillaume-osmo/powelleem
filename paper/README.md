# paper/

Draft manuscript for the `powelleem` v0.2.0 work, targeting *Journal
of Cheminformatics* (where NEEMP was published) or chemRxiv preprint.

## Files

- `paper.md`  — markdown source (~10 pages of dense scientific content)
- `paper.bib` — BibTeX bibliography

## Building

The markdown source uses inline LaTeX math (`\[...\]` and `\(...\)`).
To produce a PDF locally:

```bash
# from this directory
pandoc paper.md \
    --citeproc --bibliography paper.bib \
    --pdf-engine=xelatex \
    -o paper.pdf
```

Or render to HTML:

```bash
pandoc paper.md --citeproc --bibliography paper.bib -o paper.html
```

## Headline

`powelleem` reproduces the NEEMP CCD_gen reference parameter set and
**improves on Raček 2016 by 11.4 % mol-RMSD** on set03 (17,769
molecules), in **146 s** of wall-time vs an estimated ~10-20 h for the
original MATLAB DE+NEWUOA pipeline. The improvement comes from
replacing NEEMP's derivative-free NEWUOA polish with a trust-Newton
polish driven by the **exact analytical Hessian** of the EEM loss,
derived in closed form via two applications of the implicit function
theorem to the per-molecule linear system.
