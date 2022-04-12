import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt


def plot_model_performance():
    save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/SVG_vs_SVG_TE_vs_SPOTS_SVG_ACTP_no_new/"
    # save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/SVG_vs_SVG_TE_vs_SPOTS_SVG_ACTP_novel/"
    model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_07_04_2022_17_04/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TE/model_07_04_2022_19_33/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP/model_08_04_2022_14_55/qualitative_analysis/BEST/test_no_new_formatted/"]
    model_names = ["SVG", "SVG_TE", "SPOTS_SVG_ACTP_BEST"]
    sequence_len = 5

    test_sequence_paths = list(glob(model_locations[0] + "*/", recursive = True))
    test_sequence_paths = [i[len(model_locations[0]):] for i in test_sequence_paths]
    for folder_name in test_sequence_paths:
        print(folder_name)
        prediction_data = []
        for path in model_locations:
            gt_data = [np.load(path + folder_name + "gt_scene_time_step_" + str(i) + ".npy") for i in range(sequence_len)]
            prediction_data.append([np.load(path + folder_name + "pred_scene_time_step_" + str(i) + ".npy") for i in range(sequence_len)])

        # create folder to save:
        sequence_save_path = save_location + folder_name + "/"
        try:
            os.mkdir(sequence_save_path)
        except FileExistsError or FileNotFoundError:
            pass

        for i in range(sequence_len):
            plt.figure(1)
            plt.rc('font', size=4)
            f, axarr = plt.subplots(1, len(model_locations)+1)
            axarr[0].set_title("GT t_" + str(i))
            axarr[0].imshow(np.array(gt_data[i]))
            for index, model_name in enumerate(model_names):
                axarr[index+1].set_title(str(model_name) + " pred" + " t_" + str(sequence_len))
                axarr[index+1].imshow(np.array(prediction_data[index][i]))
            plt.savefig(sequence_save_path + "scene_time_step_" + str(i) + ".png", dpi=300)
            plt.close('all')

    # for location in model_locations

if __name__ == '__main__':
    plot_model_performance()