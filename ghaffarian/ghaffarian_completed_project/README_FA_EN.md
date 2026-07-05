
# Ghafarian Chapters 3-4 English LaTeX Completion

This project is built around the DeepSeek English file `Chapter_3_4_English.tex` and the original thesis PDF `GhafarianThesis_ActiveLearning.pdf`.

## Files

- `Ghafarian_Ch3_4_English_complete.tex`: robust master LaTeX wrapper.
- `figures/`: expected figure filenames. The files in this completed project have been populated from the original thesis PDF.
- `build_complete_latex.py`: downloads/patches the DeepSeek file and optionally extracts embedded PDF images.

## Usage

1. Put `Chapter_3_4_English.tex` in this folder, or allow the script to download it.
2. Put `GhafarianThesis_ActiveLearning.pdf` in this folder, or allow the script to download it.
3. Run:

```bash
python build_complete_latex.py
pdflatex Ghafarian_Ch3_4_English_complete.tex
pdflatex Ghafarian_Ch3_4_English_complete.tex
```

4. The completed project already includes populated figure files in `figures/`. If you regenerate the body file, keep the same figure filenames.

## Expected figure filenames

`fig3_1a.png`, `fig3_1b.png`, `fig3_1c.png`, `fig3_1d.png`, `fig3_5.png`, `fig4_1.png`, `fig4_2.png`, `fig4_3.png`, `fig4_4.png`, `fig4_5.png`, `fig4_6.png`, `fig4_7.png`, `fig4_8.png`, `fig4_9.png`, `fig4_10.png`.
