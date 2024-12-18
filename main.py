# main.py
# Main entry point for the Transformer project utilizing historical data for training and evaluation.

import torch
from preprocessing import preprocessing_main
from training_loop import training_loop_main
from eval_model import eval_transformer_main
from src.utils.debug_utils import CONSOLE_BANNER

# Set PyTorch print precision for better debug visibility
torch.set_printoptions(precision=7)

# Define device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flags to toggle training and evaluation modes
bool_train_transformer = False
bool_eval_transformer = True


def main():
    """
    Main function to control the execution flow of the program.
    """
    # Preprocessing step
    CONSOLE_BANNER("DATA PREPROCESSING")
    dataloader, processed_data = preprocessing_main(device)

    # Training step (if enabled)
    if bool_train_transformer:
        CONSOLE_BANNER("TRAINING TRANSFORMER")
        training_loop_main(device, dataloader)

    # Evaluation step (if enabled)
    if bool_eval_transformer:
        CONSOLE_BANNER("EVALUATING TRANSFORMER")
        eval_transformer_main(device, processed_data)


if __name__ == "__main__":
    # Display program start banner and system info
    CONSOLE_BANNER("PROGRAM START")
    print("© 2024 Omar Samha. All rights reserved.")
    print("PyTorch Version:", torch.__version__, "\n\n\n")

    # Execute main function
    main()

    # Display program end banner
    CONSOLE_BANNER("END OF PROGRAM")
