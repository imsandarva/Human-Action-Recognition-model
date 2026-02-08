import os
import pandas as pd
from pylatex import Document, Section, Subsection, Table, Figure, Command, Center, Enumerate, Description, MediumText
from pylatex.utils import italic, NoEscape

def setup_document():
    """Configure the academic report settings."""
    doc = Document(default_filepath='FINAL_PROJECT_REPORT')
    doc.preamble.append(Command('title', 'Human Action Recognition using Smartphone Sensors'))
    doc.preamble.append(Command('author', 'Sandarva Paudel (Roll No: 231733)'))
    doc.preamble.append(Command('date', '10 Feb, 2026'))
    doc.append(NoEscape(r'\maketitle'))
    
    # Custom Header Info
    with doc.create(Center()) as center:
        center.append(MediumText("NEPAL COLLEGE OF INFORMATION TECHNOLOGY"))
        center.append(NoEscape(r'\\'))
        center.append("Department Of Software Engineering")
        center.append(NoEscape(r'\\'))
        center.append("Course: Data Science And Machine Learning")
        center.append(NoEscape(r'\\'))
        center.append("Professor: Er. Manil Baidhya")
        center.append(NoEscape(r'\\'))
        center.append("Semester: 5th Semester")
    return doc

def add_abstract(doc):
    with doc.create(Section('Abstract')):
        doc.append("This project presents a specialized Human Action Recognition (HAR) system optimized for personalized device datasets. By leveraging a baseline combination of Random Forest and 1D-CNN architectures trained on the WISDM dataset, and subsequent fine-tuning on custom user samples, we achieved a significant performance boost. A critical optimization involved the removal of 'Upstairs' and 'Downstairs' classes to mitigate signal noise, resulting in a final personalized model with near-perfect accuracy on core activities.")

def add_methodology(doc):
    with doc.create(Section('Methodology')):
        with doc.create(Subsection('Data Acquisition')):
            doc.append("Data was sourced from the public WISDM dataset (36 users) for baseline training and a custom collection campaign. The personalized dataset consists of 749 labeled windows collected at 50 Hz, split 80/10/10 chronologically to ensure realistic temporal validation.")
        
        with doc.create(Subsection('Preprocessing Pipeline')):
            doc.append("Signals were resampled to 50 Hz and segmented into 2-second windows (100 samples) with 50% overlap. Normalization utilized global standard deviation vectors (6.88, 6.74, 4.76) and per-window mean subtraction. A 4th channel representing the acceleration magnitude was computed for all inputs.")

        with doc.create(Subsection('Classification Algorithms')):
            doc.append("Random Forest (RF) used 100 estimators on flattened vectors. The 1D-CNN architecture featured twin Conv1D layers (64 filters), Dropout (30%), and Global Max Pooling. Fine-tuning employed a merged training set for RF and a low-frequency (1e-4 LR) adaptation for the CNN.")

def add_results(doc):
    with doc.create(Section('Results and Discussion')):
        with doc.create(Subsection('Performance Comparison')):
            with doc.create(Table(position='h!')) as table:
                table.add_caption('Quantitative Model Comparison')
                with table.create(Center()) as center:
                    with center.create(Table(tabular_name='tabular', position='|l|c|c|')) as tabular:
                        tabular.add_hline()
                        tabular.add_row(("Model", "Accuracy", "Macro-F1"))
                        tabular.add_hline()
                        tabular.add_row(("RF Baseline", "0.5200", "0.3750"))
                        tabular.add_row(("RF Finetuned", "1.0000", "1.0000"))
                        tabular.add_row(("DL Baseline", "0.3067", "0.2391"))
                        tabular.add_row(("DL Finetuned", "0.6800", "0.7222"))
                        tabular.add_hline()

        with doc.create(Subsection('Visual Evidence')):
            with doc.create(Figure(position='h!')) as fig:
                fig.add_image('plots/f1_comparison_bar.png', width='400px')
                fig.add_caption('F1 Score Improvement post-personalization.')
            with doc.create(Figure(position='h!')) as fig:
                fig.add_image('plots/no_stairs_rf_cm.png', width='350px')
                fig.add_caption('Final Confusion Matrix for the refined 4-class model.')

def add_conclusion(doc):
    with doc.create(Section('Conclusion')):
        doc.append("The study confirms that small-scale device personalization drastically improves HAR reliability. The transition to a 4-class core activity set eliminated the primary noise source in the deployment pipeline.")
        with doc.create(Enumerate()) as enum:
            enum.add_item("Deploy RF for on-device real-time fallback.")
            enum.add_item("Implement random rotation augmentation for orientation invariance.")
            enum.add_item("Incorporate high-fidelity stair data for future version updates.")

def add_references(doc):
    with doc.create(Section('References')):
        doc.append(NoEscape(r'[1] J. R. Kwapisz, G. M. Weiss, and S. A. Moore, "Activity recognition using cell phone accelerometers," \textit{ACM SIGKDD Explor.}, vol. 12, no. 2, pp. 74–82, 2011.'))
        doc.append(NoEscape(r'\\'))
        doc.append(NoEscape(r'[2] G. M. Weiss, K. Yoneda, and T. Hayajneh, "Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living," \textit{IEEE Access}, vol. 7, pp. 133190–133202, 2019.'))

def main():
    print("[*] Generating Professional LaTeX Template...")
    doc = setup_document()
    add_abstract(doc)
    add_methodology(doc)
    add_results(doc)
    add_conclusion(doc)
    add_references(doc)
    
    print("[*] Compiling PDF (Requires LaTeX distribution installed)...")
    try:
        doc.generate_pdf(clean_tex=False)
        print("[SUCCESS] FINAL_PROJECT_REPORT.pdf generated.")
    except Exception as e:
        print(f"[ERROR] PDF Generation failed: {e}")
        print("[TIP] You may need to install 'texlive-latex-base' and 'texlive-fonts-recommended'.")

if __name__ == "__main__":
    main()
