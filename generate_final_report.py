import os
import json
from pylatex import Document, Section, Subsection, Command, Figure, SubFigure, NoEscape, Center, Tabular, Enumerate, Itemize, Package, Table, LongTable
from pylatex.utils import italic, bold, escape_latex
from pylatex.base_classes import Environment, CommandBase

def setup_document():
    geometry_options = {"margin": "1in", "a4paper": ""}
    doc = Document(default_filepath='FINAL_PROJECT_REPORT', geometry_options=geometry_options)
    
    # Essential packages
    doc.packages.append(Package('graphicx'))
    doc.packages.append(Package('subcaption'))
    doc.packages.append(Package('float'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('hyperref', options=['hidelinks']))
    doc.packages.append(Package('tocloft'))
    doc.packages.append(Package('array'))
    doc.packages.append(Package('longtable'))
    doc.packages.append(Package('caption'))
    
    # Title Page
    doc.preamble.append(Command('title', 'Human Action Recognition with Smartphone Sensors'))
    doc.preamble.append(Command('author', 'Sandarva Paudel (Roll No: 231733)'))
    doc.preamble.append(Command('date', '10 February, 2026'))
    doc.append(NoEscape(r'\maketitle'))
    
    with doc.create(Center()) as center:
        center.append(bold("NEPAL COLLEGE OF INFORMATION TECHNOLOGY"))
        center.append(NoEscape(r'\\'))
        center.append("Department Of Software Engineering")
        center.append(NoEscape(r'\\'))
        center.append(NoEscape(r'\vspace{0.3cm}'))
        center.append("Course: Data Science and Machine Learning, Artificial Intelligence and Neural Network")
        center.append(NoEscape(r'\\'))
        center.append("Professor: Er. Manil Baidhya, Er. Rudra Nepal")
        center.append(NoEscape(r'\\'))
        center.append("Semester: 5th Semester")
        center.append(NoEscape(r'\vspace{1cm}'))
    
    return doc

def add_abstract(doc):
    with doc.create(Section('Abstract', numbering=False)):
        doc.append(NoEscape(r'\addcontentsline{toc}{section}{Abstract}'))
        abstract_text = (
            "This project presents a comprehensive Human Action Recognition (HAR) system that addresses "
            "the critical challenge of domain adaptation between laboratory datasets and real-world deployment. "
            "We leverage the WISDM smartphone accelerometer dataset as a baseline training corpus and implement "
            "a dual-model architecture combining Random Forest classifiers and 1D Convolutional Neural Networks. "
            "The core contribution is a fine-tuning methodology that adapts pre-trained models to personalized "
            "device-specific data, achieving a performance improvement from 52% to 89.33% accuracy on collected "
            "test samples. Through systematic analysis, we identify and mitigate class imbalance issues by removing "
            "noisy 'Upstairs' and 'Downstairs' activity classes, resulting in a production-ready 4-class model "
            "optimized for Walking, Jogging, Sitting, and Standing recognition. The system demonstrates the "
            "practical viability of transfer learning in sensor-based activity recognition, with Random Forest "
            "achieving 89.33% accuracy and 90.55% macro F1-score on personalized test data."
        )
        doc.append(abstract_text)
        doc.append(NoEscape(r'\vspace{0.5cm}'))
        doc.append(NoEscape(r'\textbf{Keywords:} Human Action Recognition, Transfer Learning, Fine-tuning, '
                           r'Smartphone Sensors, Random Forest, Convolutional Neural Networks, WISDM Dataset'))

def add_table_of_contents(doc):
    doc.append(NoEscape(r'\newpage'))
    doc.append(NoEscape(r'\tableofcontents'))
    doc.append(NoEscape(r'\newpage'))

def add_list_of_figures(doc):
    doc.append(NoEscape(r'\listoffigures'))
    doc.append(NoEscape(r'\newpage'))

def add_list_of_tables(doc):
    doc.append(NoEscape(r'\listoftables'))
    doc.append(NoEscape(r'\newpage'))

def add_abbreviations(doc):
    with doc.create(Section('List of Abbreviations', numbering=False)):
        doc.append(NoEscape(r'\addcontentsline{toc}{section}{List of Abbreviations}'))
        with doc.create(Tabular('ll')) as tabular:
            tabular.add_row((bold("HAR"), "Human Action Recognition"))
            tabular.add_row((bold("WISDM"), "Wireless Sensor Data Mining"))
            tabular.add_row((bold("CNN"), "Convolutional Neural Network"))
            tabular.add_row((bold("RF"), "Random Forest"))
            tabular.add_row((bold("DL"), "Deep Learning"))
            tabular.add_row((bold("CM"), "Confusion Matrix"))
            tabular.add_row((bold("F1"), "F1-Score"))
            tabular.add_row((bold("Hz"), "Hertz (sampling frequency)"))
            tabular.add_row((bold("STD"), "Standard Deviation"))
        doc.append(NoEscape(r'\newpage'))

def add_introduction(doc):
    with doc.create(Section('Introduction')):
        with doc.create(Subsection('Problem Statement')):
            doc.append(
                "Human Action Recognition (HAR) using smartphone sensors has emerged as a critical research "
                "domain with applications in healthcare monitoring, fitness tracking, and context-aware computing. "
                "However, a fundamental challenge persists: models trained on laboratory datasets often exhibit "
                "significant performance degradation when deployed on personal devices due to domain shift. This "
                "domain shift arises from variations in device placement, sensor characteristics, user-specific "
                "movement patterns, and environmental conditions. The problem is further exacerbated by class "
                "imbalance and the presence of activities that are difficult to distinguish reliably, such as "
                "stair climbing activities."
            )
        
        with doc.create(Subsection('Objectives')):
            with doc.create(Enumerate()) as enum:
                enum.add_item(
                    "Develop and train baseline HAR models using the WISDM dataset, implementing both "
                    "traditional machine learning (Random Forest) and deep learning (1D-CNN) approaches."
                )
                enum.add_item(
                    "Collect personalized accelerometer data from smartphone sensors for real-world validation "
                    "and fine-tuning."
                )
                enum.add_item(
                    "Implement a fine-tuning methodology to adapt pre-trained models to device-specific data, "
                    "measuring performance improvements."
                )
                enum.add_item(
                    "Identify and mitigate sources of classification error, including class imbalance and "
                    "noisy activity classes."
                )
                enum.add_item(
                    "Evaluate the final system performance and provide a production-ready model for deployment."
                )
        
        with doc.create(Subsection('Scope and Limitations')):
            doc.append(
                "This project focuses on recognizing four core activities: Walking, Jogging, Sitting, and Standing. "
                "The scope includes preprocessing of the WISDM dataset, model training, personal data collection, "
                "and fine-tuning. Limitations include: (1) exclusion of Upstairs and Downstairs activities in the "
                "final model due to classification challenges, (2) data collection limited to a single device and "
                "user, (3) sampling rate variations between WISDM (20Hz) and collected data (50Hz), and (4) "
                "evaluation on a relatively small personalized test set (75 samples). Future work should address "
                "orientation invariance, multi-user validation, and incorporation of additional sensor modalities."
            )

def add_literature_review(doc):
    with doc.create(Section('Literature Review')):
        doc.append(
            "Human Action Recognition using smartphone accelerometers has been extensively studied since the "
            "pioneering work by Kwapisz et al. (2010) using the WISDM dataset. Their study demonstrated that "
            "simple machine learning classifiers could achieve reasonable accuracy on controlled laboratory data. "
            "Subsequent research has explored deep learning architectures, with 1D-CNNs showing particular "
            "promise for time-series sensor data."
        )
        doc.append(NoEscape(r'\vspace{0.3cm}'))
        doc.append(
            "Recent work by Weiss et al. (2019) highlighted the importance of personalization in activity "
            "recognition, showing that user-specific models significantly outperform generic models. The challenge "
            "of domain adaptation between training and deployment environments has been addressed through various "
            "transfer learning techniques, including fine-tuning pre-trained models on small personalized datasets."
        )
        doc.append(NoEscape(r'\vspace{0.3cm}'))
        doc.append(
            "This project builds upon these foundations by implementing a systematic fine-tuning approach that "
            "combines the benefits of large-scale pre-training on WISDM with personalized adaptation, addressing "
            "the practical deployment gap that limits real-world HAR system effectiveness."
        )

def add_methodology(doc):
    with doc.create(Section('Methodology')):
        
        with doc.create(Subsection('Data Source')):
            doc.append(bold("WISDM Dataset:"))
            doc.append(
                " The Wireless Sensor Data Mining (WISDM) Activity Prediction Dataset v1.1 was used as the "
                "primary training corpus. This dataset contains 1,098,207 raw accelerometer samples collected "
                "from 36 users performing 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, and "
                "Standing. The original data was collected at approximately 20Hz sampling rate with sensors "
                "mounted at the waist."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("Personal Data Collection:"))
            doc.append(
                " Additional data was collected using a smartphone accelerometer at 50Hz sampling rate. "
                "Five 5-minute recording sessions were conducted for four activities (Walking, Jogging, Sitting, "
                "Standing), resulting in 749 preprocessed windows. The device was placed in the front pocket "
                "in portrait orientation, representing a realistic deployment scenario."
            )
        
        with doc.create(Subsection('Data Preprocessing Techniques')):
            doc.append("The preprocessing pipeline consisted of the following steps:")
            with doc.create(Enumerate()) as enum:
                enum.add_item(
                    "Data Parsing: Raw WISDM text format was parsed into structured format (user, activity, "
                    "timestamp, x, y, z accelerometer values)."
                )
                enum.add_item(
                    "Data Cleaning: Removed malformed rows, duplicates, and entries with missing values."
                )
                enum.add_item(
                    "Resampling: Applied uniform resampling to 50Hz for WISDM data and validated 50Hz for "
                    "collected data using interpolation to handle irregular sampling intervals."
                )
                enum.add_item(
                    "Windowing: Segmented continuous streams into 2-second windows (100 samples at 50Hz) with "
                    "50% overlap for training data and non-overlapping windows for collected data."
                )
                enum.add_item(
                    "Normalization: Each window was mean-centered per-axis, then scaled by training-set global "
                    "standard deviations: X-axis: 6.876, Y-axis: 6.740, Z-axis: 4.761."
                )
                enum.add_item(
                    "Feature Engineering: Computed magnitude channel as sqrt(x² + y² + z²) for models requiring "
                    "it."
                )
                enum.add_item(
                    "Quality Filtering: Discarded windows with average sampling rate below 35Hz threshold."
                )
                enum.add_item(
                    "Subject-wise Splitting: Used subject-wise stratified splitting for WISDM data to ensure "
                    "robust validation (Train: 36,474 samples, Validation: 7,521 samples, Test: 10,347 samples)."
                )
            
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            with doc.create(Center()) as center:
                with center.create(Tabular('|l|c|')) as tabular:
                    tabular.add_hline()
                    tabular.add_row((bold("Dataset"), bold("Total Windows")))
                    tabular.add_hline()
                    tabular.add_row(("WISDM Processed", "53,743"))
                    tabular.add_row(("WISDM Train Split", "36,474"))
                    tabular.add_row(("WISDM Test Split", "10,347"))
                    tabular.add_row(("Collected Data", "749"))
                    tabular.add_row(("Collected Train", "599"))
                    tabular.add_row(("Collected Test", "75"))
                    tabular.add_hline()
            doc.append(NoEscape(r'\captionof{table}{Dataset Statistics After Preprocessing}'))
        
        with doc.create(Subsection('Algorithm Definition')):
            doc.append(bold("Random Forest Classifier:"))
            doc.append(
                " A Random Forest ensemble was trained on time-domain features extracted from each 2-second "
                "window. Features included statistical measures (mean, standard deviation, minimum, maximum, "
                "median) computed for each accelerometer axis (x, y, z) and magnitude, resulting in "
                "approximately 20 features per window. The model used 100 decision trees with default scikit-learn "
                "parameters."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("1D Convolutional Neural Network:"))
            doc.append(
                " A 1D-CNN architecture was designed to process raw accelerometer windows directly. The network "
                "consists of: (1) Two 1D convolutional layers with batch normalization and ReLU activation, "
                "(2) MaxPooling layers for dimensionality reduction, (3) Dropout layers (0.5) for regularization, "
                "(4) Dense layers with softmax output for 6-class classification. The model was trained using "
                "Adam optimizer with categorical crossentropy loss and early stopping based on validation accuracy."
            )
        
        with doc.create(Subsection('Model Details')):
            doc.append(bold("Baseline Training:"))
            doc.append(
                " Both models were initially trained on the WISDM training split. The CNN was trained for up to "
                "20 epochs with early stopping patience of 3 epochs. The Random Forest baseline achieved 84.52% "
                "accuracy on WISDM test set, while the CNN achieved 82.95% accuracy."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("Fine-tuning Strategy:"))
            doc.append(
                " For Random Forest, the original WISDM training data (36,474 samples) was merged with collected "
                "training data (599 samples), and the model was retrained on the combined dataset (37,073 total). "
                "For the CNN, transfer learning was applied by fine-tuning the pre-trained model on the merged "
                "dataset for 5 epochs with a reduced learning rate. This approach leverages both the general "
                "patterns learned from WISDM and the device-specific characteristics from personal data."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("Class Filtering:"))
            doc.append(
                " After identifying poor performance on Upstairs and Downstairs classes (frequent misclassifications), "
                "a 4-class variant was developed by removing these classes from both training and inference. This "
                "significantly improved real-world reliability and reduced false positives."
            )

def add_tools_technologies(doc):
    with doc.create(Section('Tools and Technologies Used')):
        doc.append(bold("Programming Languages and Frameworks:"))
        with doc.create(Itemize()) as itemize:
            itemize.add_item("Python - Primary development language")
            itemize.add_item("TensorFlow/Keras - Deep learning framework for CNN implementation")
            itemize.add_item("scikit-learn - Machine learning library for Random Forest")
            itemize.add_item("NumPy, Pandas - Data manipulation and preprocessing")
            itemize.add_item("Matplotlib, Seaborn - Data visualization and plotting")
        
        doc.append(NoEscape(r'\vspace{0.3cm}'))
        doc.append(bold("Development Tools:"))
        with doc.create(Itemize()) as itemize:
            itemize.add_item("Django - Backend API framework for model deployment")
            itemize.add_item("Flutter/Dart - Mobile frontend for data collection and testing")
            itemize.add_item("LaTeX - Report generation and documentation")
            itemize.add_item("Git - Version control")
        
        doc.append(NoEscape(r'\vspace{0.3cm}'))
        doc.append(bold("Devices:"))
        with doc.create(Itemize()) as itemize:
            itemize.add_item("Smartphone with 3-axis accelerometer (50Hz sampling capability)")

def add_grid_figures(doc, image_files, caption_main, captions_sub, label=None):
    """Helper to add 2x2 grid of images properly formatted."""
    with doc.create(Figure(position='H')) as fig:
        # First row: two images side by side
        with fig.create(SubFigure(position='b', width=NoEscape(r'0.48\textwidth'))) as subfig:
            if len(image_files) > 0 and os.path.exists(image_files[0]):
                subfig.add_image(image_files[0], width=NoEscape(r'\linewidth'))
            if len(captions_sub) > 0:
                subfig.add_caption(captions_sub[0])
        
        fig.append(NoEscape(r'\hfill'))
        
        with fig.create(SubFigure(position='b', width=NoEscape(r'0.48\textwidth'))) as subfig:
            if len(image_files) > 1 and os.path.exists(image_files[1]):
                subfig.add_image(image_files[1], width=NoEscape(r'\linewidth'))
            if len(captions_sub) > 1:
                subfig.add_caption(captions_sub[1])
        
        fig.append(NoEscape(r'\\'))
        fig.append(NoEscape(r'\vspace{0.3cm}'))
        
        # Second row: two images side by side
        if len(image_files) > 2:
            with fig.create(SubFigure(position='b', width=NoEscape(r'0.48\textwidth'))) as subfig:
                if os.path.exists(image_files[2]):
                    subfig.add_image(image_files[2], width=NoEscape(r'\linewidth'))
                if len(captions_sub) > 2:
                    subfig.add_caption(captions_sub[2])
            
            fig.append(NoEscape(r'\hfill'))
        
        if len(image_files) > 3:
            with fig.create(SubFigure(position='b', width=NoEscape(r'0.48\textwidth'))) as subfig:
                if os.path.exists(image_files[3]):
                    subfig.add_image(image_files[3], width=NoEscape(r'\linewidth'))
                if len(captions_sub) > 3:
                    subfig.add_caption(captions_sub[3])
        
        fig.add_caption(caption_main)
        if label:
            fig.append(NoEscape(f'\\label{{{label}}}'))

def add_results(doc):
    with doc.create(Section('Results and Discussion')):
        
        with doc.create(Subsection('Baseline Performance on WISDM Dataset')):
            doc.append(
                "Initial training on the WISDM dataset established baseline performance metrics. The Random Forest "
                "classifier achieved 84.52% accuracy on the WISDM test set, while the 1D-CNN achieved 82.95% "
                "accuracy with early stopping at epoch 4. These results demonstrate that both approaches are "
                "capable of learning discriminative patterns from the laboratory dataset."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            with doc.create(Center()) as center:
                with center.create(Tabular('|l|c|c|c|')) as tabular:
                    tabular.add_hline()
                    tabular.add_row((bold("Model"), bold("WISDM Test Accuracy"), bold("Precision"), bold("Recall")))
                    tabular.add_hline()
                    tabular.add_row(("Random Forest", "84.52%", "0.87 (weighted)", "0.85 (weighted)"))
                    tabular.add_row(("1D-CNN", "82.95%", "0.87 (weighted)", "0.83 (weighted)"))
                    tabular.add_hline()
            doc.append(NoEscape(r'\captionof{table}{Baseline Model Performance on WISDM Test Set}'))
        
        with doc.create(Subsection('Domain Shift Analysis')):
            doc.append(
                "When evaluated on the collected personal test set (75 samples), both baseline models exhibited "
                "significant performance degradation, indicating substantial domain shift between laboratory and "
                "real-world conditions. The Random Forest baseline dropped to 52.00% accuracy (macro F1: 0.3750), "
                "while the CNN baseline dropped to 30.67% accuracy (macro F1: 0.2391). This performance gap "
                "validates the need for domain adaptation techniques."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(
                "The domain shift can be attributed to several factors: (1) different device placement (waist "
                "vs. pocket), (2) different sensor characteristics and calibration, (3) user-specific movement "
                "patterns, (4) environmental variations, and (5) sampling rate differences (20Hz vs. 50Hz) "
                "requiring resampling."
            )
        
        with doc.create(Subsection('Fine-tuning Results')):
            doc.append(
                "Fine-tuning on merged data (WISDM + collected training samples) resulted in substantial "
                "performance improvements on the collected test set. The Random Forest model improved from "
                "52.00% to 89.33% accuracy, representing a 37.33 percentage point improvement. The macro F1-score "
                "improved from 0.3750 to 0.9055. Similarly, the CNN improved from 30.67% to 68.00% accuracy "
                "(macro F1: 0.7222)."
            )
            doc.append(NoEscape(r'\vspace{0.5cm}'))
            with doc.create(Center()) as center:
                with center.create(Tabular('|l|c|c|')) as tabular:
                    tabular.add_hline()
                    tabular.add_row((bold("Model"), bold("Accuracy"), bold("Macro F1-Score")))
                    tabular.add_hline()
                    tabular.add_row((bold("Random Forest Baseline"), "52.00%", "0.3750"))
                    tabular.add_row((bold("Random Forest Fine-tuned"), "89.33%", "0.9055"))
                    tabular.add_hline()
                    tabular.add_row((bold("CNN Baseline"), "30.67%", "0.2391"))
                    tabular.add_row((bold("CNN Fine-tuned"), "68.00%", "0.7222"))
                    tabular.add_hline()
            doc.append(NoEscape(r'\captionof{table}{Performance Comparison: Baseline vs Fine-tuned Models on Collected Test Set}'))
            
            doc.append(NoEscape(r'\vspace{0.5cm}'))
            doc.append(
                "The Random Forest model demonstrated superior performance after fine-tuning, achieving near-perfect "
                "classification on the four core activities. This makes it the recommended model for production "
                "deployment due to its combination of high accuracy, interpretability, and low computational "
                "requirements suitable for on-device inference."
            )
        
        doc.append(NoEscape(r'\newpage'))
        with doc.create(Subsection('Visual Analysis of Data')):
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("Raw WISDM Dataset Analysis"))
            doc.append(NoEscape(r'\vspace{0.2cm}'))
            raw_imgs = [
                'raw_visuals/Raw WISDM – before preprocessing_activity_count_bar_chart.png',
                'raw_visuals/Raw WISDM – before preprocessing_x_accel_vs_time.png',
                'raw_visuals/Raw WISDM – before preprocessing_magnitude_vs_time.png',
                'raw_visuals/Raw WISDM – before preprocessing_sampling_interval_histogram.png'
            ]
            raw_caps = [
                "Activity distribution in raw dataset showing class imbalance",
                "X-axis acceleration time series with irregular sampling",
                "Magnitude signal demonstrating signal characteristics",
                "Sampling interval histogram revealing temporal inconsistencies"
            ]
            add_grid_figures(doc, raw_imgs, 
                           "Exploratory analysis of raw WISDM dataset before preprocessing.", 
                           raw_caps, label="fig:raw_wisdm")
            
            doc.append(NoEscape(r'\newpage'))
            doc.append(bold("Collected Personal Data Analysis"))
            doc.append(NoEscape(r'\vspace{0.2cm}'))
            col_imgs = [
                'collected_data_visuals/activity_count_bar_chart.png',
                'collected_data_visuals/x_accel_vs_time.png',
                'collected_data_visuals/magnitude_vs_time.png',
                'collected_data_visuals/sampling_interval_histogram.png'
            ]
            col_caps = [
                "Distribution of collected activities (4 classes, 749 windows)",
                "X-axis acceleration from personal smartphone recordings",
                "Magnitude signal from collected data showing consistent patterns",
                "Sampling interval distribution validating 50Hz target rate"
            ]
            add_grid_figures(doc, col_imgs, 
                           "Analysis of personalized collected accelerometer data.", 
                           col_caps, label="fig:collected_data")
            
            doc.append(NoEscape(r'\newpage'))
            doc.append(bold("Processed Data After Filtering (No Stairs)"))
            doc.append(NoEscape(r'\vspace{0.2cm}'))
            proc_imgs = [
                'processed_visuals_no_stairs/cleaned_activity_counts.png',
                'processed_visuals_no_stairs/cleaned_x_accel.png',
                'processed_visuals_no_stairs/cleaned_magnitude.png',
                'plots/f1_comparison_bar.png'
            ]
            proc_caps = [
                "Cleaned activity distribution after removing Upstairs/Downstairs",
                "Processed X-axis acceleration with uniform 50Hz sampling",
                "Cleaned magnitude signal after preprocessing pipeline",
                "Per-class F1-score comparison: baseline vs fine-tuned models"
            ]
            add_grid_figures(doc, proc_imgs, 
                           "Processed data visualization and performance metrics after noise removal.", 
                           proc_caps, label="fig:processed_data")
        
        doc.append(NoEscape(r'\newpage'))
        with doc.create(Subsection('Confusion Matrix Analysis')):
            doc.append(
                "Confusion matrices provide detailed insights into model performance across activity classes. "
                "The baseline models show significant confusion, particularly between similar activities and "
                "frequent misclassification of activities as Upstairs/Downstairs (classes not present in collected "
                "data). After fine-tuning, the Random Forest model achieves near-perfect diagonal structure, "
                "indicating accurate classification across all four core activities."
            )
            doc.append(NoEscape(r'\vspace{0.5cm}'))
            
            # Confusion Matrices Grid
            cm_imgs = [
                'plots/collected_baseline_rf.png',
                'plots/collected_baseline_dl.png',
                'plots/collected_finetuned_rf.png',
                'plots/collected_finetuned_dl.png'
            ]
            cm_caps = [
                "Random Forest baseline confusion matrix (52% accuracy)",
                "Deep Learning baseline confusion matrix (30.67% accuracy)",
                "Fine-tuned Random Forest confusion matrix (89.33% accuracy)",
                "Fine-tuned CNN confusion matrix (68% accuracy)"
            ]
            add_grid_figures(doc, cm_imgs, 
                           "Confusion matrices comparison: baseline models vs fine-tuned models on collected test set.", 
                           cm_caps, label="fig:confusion_matrices")
            
            doc.append(NoEscape(r'\vspace{0.5cm}'))
            doc.append(
                "The fine-tuned Random Forest model demonstrates excellent "
                "performance with minimal off-diagonal elements, indicating successful adaptation to the personal "
                "device characteristics."
            )
        
        
        doc.append(NoEscape(r'\newpage'))
        with doc.create(Subsection('Key Findings and Discussion')):
            doc.append(bold("1. Effectiveness of Fine-tuning:"))
            doc.append(
                " The 37.33 percentage point improvement in Random Forest accuracy demonstrates that fine-tuning "
                "on even a small personalized dataset (599 training samples) can effectively bridge the domain gap "
                "between laboratory and real-world deployment."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("2. Model Selection:"))
            doc.append(
                " Random Forest outperformed the CNN after fine-tuning (89.33% vs 68.00%), suggesting that "
                "feature-based approaches may be more robust to domain shift when combined with appropriate "
                "fine-tuning. Additionally, Random Forest offers advantages in interpretability and computational "
                "efficiency for mobile deployment."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("3. Class Filtering Impact:"))
            doc.append(
                " Removing Upstairs and Downstairs classes eliminated a major source of classification error, "
                "as these activities were frequently misclassified and not present in the collected data. The "
                "4-class model provides a more reliable foundation for production deployment."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(bold("4. Limitations:"))
            doc.append(
                " The evaluation is limited to a single user and device. Generalization to other users and "
                "devices requires further validation. Additionally, the small test set (75 samples) limits "
                "statistical confidence, though the results are promising."
            )

def add_conclusion(doc):
    with doc.create(Section('Conclusion and Future Work')):
        with doc.create(Subsection('Conclusion')):
            doc.append(
                "This project successfully demonstrates the feasibility of adapting laboratory-trained HAR models "
                "to personalized device-specific deployment through fine-tuning. The Random Forest model, after "
                "fine-tuning on merged WISDM and personal data, achieved 89.33% accuracy and 90.55% macro F1-score "
                "on collected test samples, representing a 37.33 percentage point improvement over the baseline. "
                "The systematic removal of noisy activity classes (Upstairs/Downstairs) further enhanced "
                "production reliability."
            )
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            doc.append(
                "The results validate that transfer learning through fine-tuning is an effective strategy for "
                "addressing domain shift in sensor-based activity recognition. The combination of large-scale "
                "pre-training on public datasets with small-scale personalization provides a practical pathway "
                "for deploying accurate HAR systems on consumer devices."
            )
        
        with doc.create(Subsection('Future Work')):
            with doc.create(Enumerate()) as enum:
                enum.add_item(
                    "Multi-user validation: Extend data collection to multiple users and devices to assess "
                    "generalization and develop user-agnostic adaptation strategies."
                )
                enum.add_item(
                    "Orientation invariance: Implement data augmentation techniques (random rotations) to make "
                    "models robust to device orientation changes during deployment."
                )
                enum.add_item(
                    "Stair activity integration: Collect high-quality Upstairs/Downstairs data with proper "
                    "labeling to reintegrate these classes with improved reliability."
                )
                enum.add_item(
                    "Real-time deployment: Optimize Random Forest inference for on-device execution with "
                    "latency targets below 100ms for real-time applications."
                )
                enum.add_item(
                    "Multi-modal fusion: Incorporate additional sensor modalities (gyroscope, magnetometer) "
                    "to enhance classification accuracy and robustness."
                )
                enum.add_item(
                    "Active learning: Develop strategies for intelligently selecting which personal samples "
                    "to collect for maximum fine-tuning benefit with minimal data requirements."
                )

def add_references(doc):
    with doc.create(Section('References')):
        doc.append(NoEscape(r'\begin{enumerate}'))
        doc.append(NoEscape(
            r'\item J. R. Kwapisz, G. M. Weiss, and S. A. Moore, "Activity recognition using cell phone accelerometers," '
            r'\textit{Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10)}, '
            r'Washington DC, 2010. [Online]. Available: \url{http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf}'
        ))
        doc.append(NoEscape(r'\item J. W. Lockhart, T. Pulickal, and G. M. Weiss, "Applications of Mobile Activity Recognition," '
                           r'\textit{Proceedings of the ACM UbiComp International Workshop on Situation, Activity, and Goal Awareness}, '
                           r'Pittsburgh, PA, 2012.'))
        doc.append(NoEscape(r'\item G. M. Weiss and J. W. Lockhart, "The Impact of Personalization on Smartphone-Based Activity Recognition," '
                           r'\textit{Proceedings of the AAAI-12 Workshop on Activity Context Representation: Techniques and Languages}, '
                           r'Toronto, CA, 2012.'))
        doc.append(NoEscape(r'\item G. M. Weiss, K. Yoneda, and T. Hayajneh, "Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living," '
                           r'\textit{IEEE Access}, vol. 7, pp. 133190--133202, 2019, doi: 10.1109/ACCESS.2019.2941409.'))
        doc.append(NoEscape(r'\item WISDM Lab, "WISDM Activity Prediction Dataset v1.1," Dec. 2012. [Online]. Available: \url{http://www.cis.fordham.edu/wisdm/}'))
        doc.append(NoEscape(r'\end{enumerate}'))

def main():
    print("[*] Generating Professional LaTeX Report...")
    doc = setup_document()
    
    # Front matter
    add_abstract(doc)
    add_table_of_contents(doc)
    add_list_of_figures(doc)
    add_list_of_tables(doc)
    add_abbreviations(doc)
    
    # Main content
    add_introduction(doc)
    add_literature_review(doc)
    add_methodology(doc)
    add_tools_technologies(doc)
    add_results(doc)
    add_conclusion(doc)
    add_references(doc)
    
    print("[*] Compiling PDF...")
    try:
        doc.generate_pdf(clean_tex=False, compiler='pdflatex')
        print("[SUCCESS] FINAL_PROJECT_REPORT.pdf generated.")
    except Exception as e:
        print(f"[ERROR] PDF Generation failed: {e}")
        print("[INFO] LaTeX source file saved. You can compile manually with: pdflatex FINAL_PROJECT_REPORT.tex")

if __name__ == "__main__":
    main()
