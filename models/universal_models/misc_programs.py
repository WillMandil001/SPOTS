import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt


def plot_model_performance():
    save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/OCC_SVG_vs_SVTG_SE_vs_SPOTSACTP_vs_SVG_TC_vs_SVG_TC_TE_no_new/"
    # save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/SVG_vs_SVG_TE_vs_SPOTS_SVG_ACTP_novel/"
    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_07_04_2022_17_04/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TE/model_07_04_2022_19_33/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP/model_08_04_2022_14_55/qualitative_analysis/BEST/test_no_new_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_11_04_2022_16_44/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TE/model_11_04_2022_18_53/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP/model_11_04_2022_22_09/qualitative_analysis/BEST/test_novel_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_19_04_2022_10_25/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC/model_19_04_2022_11_22/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_TE/model_19_04_2022_13_02/qualitative_analysis/test_novel_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/VG/model_18_04_2022_10_50/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/VG_MMMM/model_18_04_2022_12_09/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_VG_ACTP/model_18_04_2022_14_36/qualitative_analysis/BEST/test_no_new_formatted/"]


    model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_occ/model_22_04_2022_10_52/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVTG_SE_occ/model_22_04_2022_13_07/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP_occ/model_22_04_2022_16_02/qualitative_analysis/BEST/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_occ/model_24_04_2022_16_39/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_TE_occ/model_24_04_2022_18_59/qualitative_analysis/test_no_new_formatted/"]

    model_names = ["SVG_occ", "SVTG_SE_occ", "SPOTS_SVG_ACTP_occ", "SVG_TC_occ", "SVG_TC_TE_occ"]
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

def plot_training_scores():
    svg = np.load("saved_models/SVG/model_19_04_2022_10_25/plot_validation_loss.npy")
    svg_tc = np.load("saved_models/SVG_TC/model_19_04_2022_11_22/plot_validation_loss.npy")
    svg_tc_te = np.load("saved_models/SVG_TC_TE/model_19_04_2022_13_02/plot_validation_loss.npy")

    svg = smooth_func(svg, 8)[4:-4]
    svg_tc = smooth_func(svg_tc, 8)[4:-4]
    svg_tc_te = smooth_func(svg_tc_te, 8)[4:-4]

    plt.plot(svg, label="SVG")
    plt.plot(svg_tc, label="SVG_TC")
    plt.plot(svg_tc_te, label="SVG_TC_TE")
    plt.legend()

    plt.title("Validation training MAE", loc='center')
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE")

    plt.show()

def smooth_func(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == '__main__':
    # plot_training_scores()
    plot_model_performance()