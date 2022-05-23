import os

import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import smtplib, ssl


def plot_model_performance():
    # save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/OCC_SVG_vs_SVTG_SE_vs_SPOTSACTP_vs_SVG_TC_vs_SVG_TC_TE_no_new/"
    save_location = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/SVG_vs_SVTG_SE_vs_SPOTS_SVG_ACTP/corrected_no_new/"
    model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_07_04_2022_17_04/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TE/model_07_04_2022_19_33/qualitative_analysis/test_no_new_formatted/",
                       "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP/model_08_04_2022_14_55/qualitative_analysis/BEST/test_no_new_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_11_04_2022_16_44/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TE/model_11_04_2022_18_53/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP/model_11_04_2022_22_09/qualitative_analysis/BEST/test_novel_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG/model_19_04_2022_10_25/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC/model_19_04_2022_11_22/qualitative_analysis/test_novel_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_TE/model_19_04_2022_13_02/qualitative_analysis/test_novel_formatted/"]

    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/VG/model_18_04_2022_10_50/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/VG_MMMM/model_18_04_2022_12_09/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_VG_ACTP/model_18_04_2022_14_36/qualitative_analysis/BEST/test_no_new_formatted/"]


    # model_locations = ["/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_occ/model_22_04_2022_10_52/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVTG_SE_occ/model_22_04_2022_13_07/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SPOTS_SVG_ACTP_occ/model_22_04_2022_16_02/qualitative_analysis/BEST/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_occ/model_24_04_2022_16_39/qualitative_analysis/test_no_new_formatted/",
    #                    "/home/user/Robotics/SPOTS/models/universal_models/saved_models/SVG_TC_TE_occ/model_24_04_2022_18_59/qualitative_analysis/test_no_new_formatted/"]

    model_names = ["SVG", "SVTG", "SPOTS_SVG_ACTP"]
    sequence_len = 5

    test_sequence_paths = list(glob(model_locations[0] + "*/", recursive = True))
    test_sequence_paths = [i[len(model_locations[0]):] for i in test_sequence_paths]
    for folder_name in test_sequence_paths:
        print(folder_name)
        prediction_data = []
        for path in model_locations:
            # gt_data = [np.rot90(np.load(path + folder_name + "gt_scene_time_step_" + str(i) + ".npy"), 3) for i in range(sequence_len)]
            # prediction_data.append([np.rot90(np.load(path + folder_name + "pred_scene_time_step_" + str(i) + ".npy"), 3) for i in range(sequence_len)])
            gt_data = [np.rot90(np.load(path + folder_name + "gt_scene_time_step_" + str(i) + ".npy"), 1) for i in range(sequence_len)]
            prediction_data.append([np.rot90(np.load(path + folder_name + "pred_scene_time_step_" + str(i) + ".npy"), 1) for i in range(sequence_len)])



        # create folder to save:
        sequence_save_path = save_location + folder_name # + "/"
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
            axarr[0].invert_yaxis()
            for index, model_name in enumerate(model_names):
                axarr[index+1].set_title(str(model_name) + " pred" + " t_" + str(sequence_len))
                axarr[index+1].imshow(np.array(prediction_data[index][i]))
                axarr[index+1].invert_yaxis()
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



def send_email():
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "willowmandil@gmail.com"  # Enter your address
    receiver_email = "willmandil@yahoo.co.uk"  # Enter receiver address
    password = input("Type your password and press enter: ")
    message = """\
    Subject: Hi there
    
    This message is sent from Python."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

def find_marker():
    data_svg = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/MARKED_100p_for_testing/SVG/test_formatted_100p"
    files = glob(data_svg + '/*')

    for file in files:
        for timestep in range(5):
            GT_image = np.load(file + "/gt_scene_time_step_" + str(timestep) + ".npy")
            PR_image = np.load(file + "/pred_scene_time_step_" + str(timestep) + ".npy")
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(cv2.cvtColor(np.transpose(GT_image, (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[1].imshow(cv2.cvtColor(np.transpose(PR_image, (1, 0, 2)), cv2.COLOR_BGR2RGB))
            plt.show()
            # cv2.imshow(GT_image)
            # cv2.waitKey (0)
            # cv2.destroyAllWindows ()

def plot_learning_rate():
    SVG_Scene_MAE = [0.0466177761554718, 0.071629099547863, 0.09148991107940674, 0.10915915668010712, 0.1251652091741562, 0.13833823800086975, 0.14938923716545105, 0.1591702401638031, 0.16623347997665405, 0.17447882890701294, 0.18119560182094574, 0.1870306432247162, 0.1917666643857956, 0.19600194692611694, 0.20061945915222168, 0.2026386708021164, 0.20584969222545624, 0.20722973346710205, 0.20977362990379333, 0.2097768783569336, 0.210534006357193, 0.21169309318065643, 0.21230363845825195, 0.21269699931144714, 0.21336081624031067, 0.21417081356048584]
    SVG_TC_MAE = [0.03691549226641655, 0.0564313605427742, 0.07181404531002045, 0.08534538745880127, 0.09797018766403198, 0.10951082408428192, 0.12093951553106308, 0.13245731592178345, 0.1416510045528412, 0.15125924348831177, 0.15928617119789124, 0.16680900752544403, 0.17293667793273926, 0.17825572192668915, 0.18324744701385498, 0.18608447909355164, 0.18942444026470184, 0.19056352972984314, 0.19259417057037354, 0.19207653403282166, 0.1925053894519806, 0.1935112178325653, 0.19454696774482727, 0.19607028365135193, 0.1979891061782837, 0.20072846114635468]
    SVG_TC_TE_MAE = [0.0420006662607193, 0.06228462979197502, 0.07838231325149536, 0.09330444782972336, 0.1053994745016098, 0.11735483258962631, 0.1291227787733078, 0.14133445918560028, 0.15189819037914276, 0.16301128268241882, 0.17140644788742065, 0.17888635396957397, 0.1845865249633789, 0.18941795825958252, 0.19387933611869812, 0.19641511142253876, 0.1991431713104248, 0.20080390572547913, 0.20271548628807068, 0.20183750987052917, 0.20172473788261414, 0.2028701901435852, 0.2041417807340622, 0.2061464488506317, 0.2087758630514145, 0.2127300202846527]

    SVG_PSNR = [71.84066009521484, 67.94094848632812, 65.56534576416016, 63.72071838378906, 62.37415313720703, 61.42094039916992, 60.728755950927734, 60.19895935058594, 59.808597564697266, 59.456581115722656, 59.17095947265625, 58.92402267456055, 58.73390197753906, 58.57050323486328, 58.42879867553711, 58.34234619140625, 58.24812316894531, 58.19115447998047, 58.12614822387695, 58.10360336303711, 58.06543731689453, 58.02297592163086, 57.98823928833008, 57.95827865600586, 57.92713928222656, 57.894371032714844]
    SVG_TC_PSNR = [73.97164154052734, 69.87690734863281, 67.2697982788086, 65.33390808105469, 63.97303009033203, 62.915184020996094, 62.031455993652344, 61.323524475097656, 60.77541732788086, 60.2811393737793, 59.883827209472656, 59.54764938354492, 59.283145904541016, 59.06277084350586, 58.88447189331055, 58.76544952392578, 58.669898986816406, 58.63083267211914, 58.59403991699219, 58.60710525512695, 58.601566314697266, 58.58694076538086, 58.559871673583984, 58.51749801635742, 58.44697952270508, 58.352294921875]
    SVG_TC_TE_PSNR = [72.60179901123047, 68.75438690185547, 66.40974426269531, 64.64549255371094, 63.437171936035156, 62.42919921875, 61.602508544921875, 60.87432098388672, 60.30512237548828, 59.812355041503906, 59.43073272705078, 59.1119384765625, 58.85432815551758, 58.65846252441406, 58.51183319091797, 58.41593933105469, 58.34831619262695, 58.31243133544922, 58.2955322265625, 58.3221435546875, 58.336063385009766, 58.31744384765625, 58.27785873413086, 58.21900177001953, 58.137916564941406, 58.01148223876953]

    SVG_SSIM = [0.49362701177597046, 0.4835270941257477, 0.4788370728492737, 0.4782305657863617, 0.4788242280483246, 0.47795602679252625, 0.4780677258968353, 0.4742778539657593, 0.47628143429756165, 0.4653712809085846, 0.46263813972473145, 0.46157896518707275, 0.4615640342235565, 0.46061626076698303, 0.45229512453079224, 0.45538726449012756, 0.4498891830444336, 0.45504575967788696, 0.44755953550338745, 0.4555974304676056, 0.4591761827468872, 0.4601968824863434, 0.4626043140888214, 0.46679598093032837, 0.4688984751701355, 0.4686070680618286]
    SVG_TC_SSIM = [0.5004723072052002, 0.4874367117881775, 0.48333317041397095, 0.48288992047309875, 0.4818359315395355, 0.4802532196044922, 0.48085615038871765, 0.47590070962905884, 0.4760681986808777, 0.4703272879123688, 0.4702160060405731, 0.465160995721817, 0.46008116006851196, 0.4538664221763611, 0.44545191526412964, 0.44427788257598877, 0.4359601140022278, 0.43780988454818726, 0.43103229999542236, 0.4368068277835846, 0.44072407484054565, 0.44174203276634216, 0.4439426064491272, 0.44569483399391174, 0.44661852717399597, 0.4466153681278229]
    SVG_TC_TE_SSIM = [0.49713340401649475, 0.48664674162864685, 0.48215368390083313, 0.48156678676605225, 0.48124054074287415, 0.48122888803482056, 0.48130539059638977, 0.47869738936424255, 0.4801176190376282, 0.4715346097946167, 0.47149068117141724, 0.46632081270217896, 0.4620968997478485, 0.45505213737487793, 0.4438017010688782, 0.44014614820480347, 0.43256112933158875, 0.43315109610557556, 0.426286518573761, 0.4322064518928528, 0.43760403990745544, 0.43923893570899963, 0.44082212448120117, 0.44230765104293823, 0.44351086020469666, 0.4426700472831726]


    f, axarr = plt.subplots (1, 3)
    axarr[0].set_ylabel ('MAE')
    axarr[0].set_xlabel ('Prediction Step')
    axarr[0].plot([i for i in range(len(SVG_Scene_MAE))], SVG_Scene_MAE, label="SVG")
    axarr[0].plot([i for i in range(len(SVG_TC_MAE))], SVG_TC_MAE, label="SVG-TC")
    axarr[0].plot([i for i in range(len(SVG_TC_TE_MAE))], SVG_TC_TE_MAE, label="SVG-TC-TE")
    axarr[0].legend()

    axarr[1].set_ylabel ('PSNR')
    axarr[1].set_xlabel ('Prediction Step')
    axarr[1].plot([i for i in range(len(SVG_PSNR))], SVG_PSNR, label="SVG")
    axarr[1].plot([i for i in range(len(SVG_TC_PSNR))], SVG_TC_PSNR, label="SVG-TC")
    axarr[1].plot([i for i in range(len(SVG_TC_TE_PSNR))], SVG_TC_TE_PSNR, label="SVG-TC-TE")
    axarr[1].legend ()

    axarr[2].set_ylabel ('SSIM')
    axarr[2].set_xlabel ('Prediction Step')
    axarr[2].plot([i for i in range(len(SVG_SSIM))], SVG_SSIM, label="SVG")
    axarr[2].plot([i for i in range(len(SVG_TC_SSIM))], SVG_TC_SSIM, label="SVG-TC")
    axarr[2].plot([i for i in range(len(SVG_TC_TE_SSIM))], SVG_TC_TE_SSIM, label="SVG-TC-TE")
    axarr[2].legend ()

    plt.show()


def plot_edge_case_qualitative():
    SVG_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVG_100p/test_edge_case_100p/"
    SVG_TE_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVG_TE_100p/test_edge_case_100p/"
    SVTG_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVTG_100p/test_edge_case_100p/"
    SVTG_large_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVTG_large_100p/test_edge_case_100p/"
    SPOTS_SP_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SPOTS_SP_100p/test_edge_case_100p/"
    SPOTS_STP_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SPOTS_STP_100p/test_edge_case_100p/"

    for i in range(88):
        GT = []
        pred_SVG_100p = []
        pred_SVG_TE_100p = []
        pred_SVTG_100p = []
        pred_SVTG_large_100p = []
        pred_SPOTS_SP_100p = []
        pred_SPOTS_STP_100p = []

        f, axarr = plt.subplots (7, 5)
        for timestep in range(5):
            GT.append(np.load(SVG_100p + "batch_0sub_batch_" + str(i) + "/gt_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVG_100p.append(np.load(SVG_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVG_TE_100p.append(np.load(SVG_TE_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVTG_100p.append(np.load(SVTG_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVTG_large_100p.append(np.load(SVTG_large_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SPOTS_SP_100p.append(np.load(SPOTS_SP_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SPOTS_STP_100p.append(np.load(SPOTS_STP_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))

            axarr[0,timestep].set_title("t=" + str(timestep), fontsize=6)
            axarr[0,timestep].imshow (cv2.cvtColor (np.transpose (GT[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[1,timestep].imshow (cv2.cvtColor (np.transpose (pred_SVG_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[2,timestep].imshow (cv2.cvtColor (np.transpose (pred_SVG_TE_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[3,timestep].imshow (cv2.cvtColor (np.transpose (pred_SVTG_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[4,timestep].imshow (cv2.cvtColor (np.transpose (pred_SVTG_large_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[5,timestep].imshow (cv2.cvtColor (np.transpose (pred_SPOTS_SP_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[6,timestep].imshow (cv2.cvtColor (np.transpose (pred_SPOTS_STP_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))



        axarr[0, 0].set_ylabel ('GT', fontsize=6)
        axarr[1, 0].set_ylabel ('SVG', fontsize=6)
        axarr[2, 0].set_ylabel ('SVG-TE', fontsize=6)
        axarr[3, 0].set_ylabel ('SVTG', fontsize=6)
        axarr[4, 0].set_ylabel ('SVTG-large', fontsize=6)
        axarr[5, 0].set_ylabel ('SPOTS-SP', fontsize=6)
        axarr[6, 0].set_ylabel ('SPOTS-STP', fontsize=6)


        for ii in range(len(axarr)):
            for j in range(len(axarr[0])):
                axarr[ii,j].set_xticks ([])
                axarr[ii,j].set_yticks ([])
        # plt.tight_layout ()
        wspace = -0.8  # the amount of width reserved for blank space between subplots
        hspace = 0.05  # the amount of height reserved for white space between subplots
        plt.subplots_adjust (left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        # plt.show()
        plt.savefig("edge_cases/batch_0sub_batch_" + str(i), dpi = 500)

def plot_edge_case_t4_qualitative():
    SVG_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVG_100p/test_edge_case_100p/"
    SVG_TE_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVG_TE_100p/test_edge_case_100p/"
    SVTG_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVTG_100p/test_edge_case_100p/"
    SVTG_large_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SVTG_large_100p/test_edge_case_100p/"
    SPOTS_SP_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SPOTS_SP_100p/test_edge_case_100p/"
    SPOTS_STP_100p = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_Cases/SPOTS_STP_100p/test_edge_case_100p/"

    for i in range(88):
        GT = []
        pred_SVG_100p = []
        pred_SVG_TE_100p = []
        pred_SVTG_100p = []
        pred_SVTG_large_100p = []
        pred_SPOTS_SP_100p = []
        pred_SPOTS_STP_100p = []

        f, axarr = plt.subplots (1, 7)
        for timestep in range(5):
            GT.append(np.load(SVG_100p + "batch_0sub_batch_" + str(i) + "/gt_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVG_100p.append(np.load(SVG_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVG_TE_100p.append(np.load(SVG_TE_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVTG_100p.append(np.load(SVTG_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SVTG_large_100p.append(np.load(SVTG_large_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SPOTS_SP_100p.append(np.load(SPOTS_SP_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))
            pred_SPOTS_STP_100p.append(np.load(SPOTS_STP_100p + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"))

        axarr[0].set_ylabel("t=" + str(timestep), fontsize=6)
        axarr[0].imshow (cv2.cvtColor (np.transpose (GT[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[1].imshow (cv2.cvtColor (np.transpose (pred_SVG_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[2].imshow (cv2.cvtColor (np.transpose (pred_SVG_TE_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[3].imshow (cv2.cvtColor (np.transpose (pred_SVTG_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[4].imshow (cv2.cvtColor (np.transpose (pred_SVTG_large_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[5].imshow (cv2.cvtColor (np.transpose (pred_SPOTS_SP_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))
        axarr[6].imshow (cv2.cvtColor (np.transpose (pred_SPOTS_STP_100p[timestep], (1, 0, 2)), cv2.COLOR_BGR2RGB))

        axarr[0].set_title('GT', fontsize=6)
        axarr[1].set_title('SVG', fontsize=6)
        axarr[2].set_title('SVG-TE', fontsize=6)
        axarr[3].set_title('SVTG', fontsize=6)
        axarr[4].set_title('SVTG-large', fontsize=6)
        axarr[5].set_title('SPOTS-SP', fontsize=6)
        axarr[6].set_title('SPOTS-STP', fontsize=6)

        for ii in range(len(axarr)):
                axarr[ii].set_xticks ([])
                axarr[ii].set_yticks ([])
        # plt.tight_layout ()
        # wspace = -0.8  # the amount of width reserved for blank space between subplots
        # hspace = 0.05  # the amount of height reserved for white space between subplots
        # plt.subplots_adjust (left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        # plt.show()
        plt.savefig("edge_cases/just_t4/batch_0sub_batch_" + str(i), dpi = 500)

def plot_edge_case_scene_synthesis_qualitative():
    SVG = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_seq_per_trial/SVG/test_edge_case_trail_trial_per_sequence/"
    SVG_TC = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_seq_per_trial/SVG-TC/test_edge_case_trail_trial_per_sequence/"
    SVG_TC_TE = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_seq_per_trial/SVG-TC-TE/test_edge_case_trail_trial_per_sequence/"

    for i in range (4):
        GT = []
        pred_SVG = []
        pred_SVG_TC = []
        pred_SVG_TC_TE = []

        f, axarr = plt.subplots (4, 7)
        for index, timestep in enumerate(range(0, 26, 4)):
            axarr[0,index].set_title("t=" + str(timestep), fontsize=6)
            axarr[0,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG + "batch_0sub_batch_" + str(i) + "/gt_scene_time_step_" + str(timestep) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[1,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[2,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG_TC + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[3,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG_TC_TE + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(timestep) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))

        axarr[0, 0].set_ylabel ('GT', fontsize=6)
        axarr[1, 0].set_ylabel ('SVG', fontsize=6)
        axarr[2, 0].set_ylabel ('SVG_TC', fontsize=6)
        axarr[3, 0].set_ylabel ('SVG_TC-TE', fontsize=6)

        for ii in range(len(axarr)):
            for j in range(len(axarr[0])):
                axarr[ii,j].set_xticks ([])
                axarr[ii,j].set_yticks ([])
        # plt.tight_layout ()
        # wspace = -0.8  # the amount of width reserved for blank space between subplots
        # hspace = 0.05  # the amount of height reserved for white space between subplots
        # plt.subplots_adjust (left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        # plt.show()
        plt.savefig("edge_cases/scene_synthesis/batch_0sub_batch_" + str(i), dpi = 500)


def plot_edge_case_reduced_data_qualitative():
    SVG    = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_reduced_dataset_size/SVG/"
    SVG_TE = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_reduced_dataset_size/SVG-TE/"
    SVTG   = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_reduced_dataset_size/SVTG/"
    SPOTS  = "/home/user/Robotics/SPOTS/models/universal_models/saved_models/comparison_plots/Edge_cases_reduced_dataset_size/SPOTS_STP/"

    for i in range (88):
        GT = []
        pred_SVG = []
        pred_SVG_TC = []
        pred_SVG_TC_TE = []

        f, axarr = plt.subplots (5, 5)
        for index, datasize in enumerate(range(20, 120, 20)):
            axarr[0,index].set_title(str(datasize) + "%", fontsize=6)
            axarr[0,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG + str(datasize) + "p/test_edge_case_100p/" + "batch_0sub_batch_" + str(i) + "/gt_scene_time_step_" + str(3) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[1,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG + str(datasize) + "p/test_edge_case_100p/" + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(3) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[2,index].imshow (cv2.cvtColor (np.transpose (np.load(SVG_TE + str(datasize) + "p/test_edge_case_100p/" + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(3) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[3,index].imshow (cv2.cvtColor (np.transpose (np.load(SVTG + str(datasize) + "p/test_edge_case_100p/" + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(3) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))
            axarr[4,index].imshow (cv2.cvtColor (np.transpose (np.load(SPOTS + str(datasize) + "p/test_edge_case_100p/" + "batch_0sub_batch_" + str(i) + "/pred_scene_time_step_" + str(3) + ".npy"), (1, 0, 2)), cv2.COLOR_BGR2RGB))

        axarr[0, 0].set_ylabel ('GT', fontsize=6)
        axarr[1, 0].set_ylabel ('SVG', fontsize=6)
        axarr[2, 0].set_ylabel ('SVG-TE', fontsize=6)
        axarr[3, 0].set_ylabel ('SVTG', fontsize=6)
        axarr[4, 0].set_ylabel ('SPOTS-STP', fontsize=6)

        for ii in range(len(axarr)):
            for j in range(len(axarr[0])):
                axarr[ii,j].set_xticks ([])
                axarr[ii,j].set_yticks ([])
        # plt.tight_layout ()
        wspace = -0.6  # the amount of width reserved for blank space between subplots
        hspace = 0.05  # the amount of height reserved for white space between subplots
        plt.subplots_adjust (left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
        # plt.show()
        plt.savefig("edge_cases/edge_case_reduced_data/batch_0sub_batch_" + str(i), dpi = 500)

def plot_reduced_data():
    SVG = [0.01082, 0.01092, 0.01133, 0.01208, 0.01349]
    SVG_TE = [0.01058, 0.01080, 0.01121, 0.01189, 0.01385]
    SVTG = [0.01242, 0.01377, 0.01497, 0.01558, 0.01839]
    SPOTS_STP = [0.01066, 0.01128, 0.01171, 0.01234, 0.01456]

    data = [SVG, SVG_TE, SVTG, SPOTS_STP]
    data_name = ["SVG", "SVG-TE", "SVTG", "SPOTS-STP"]

    SVG = [i - SVG[0] for i in SVG]
    SVG_TE = [i - SVG_TE[0] for i in SVG_TE]
    SVTG = [i - SVTG[0] for i in SVTG]
    SPOTS_STP = [i - SPOTS_STP[0] for i in SPOTS_STP]

    f, axarr = plt.subplots(1, 1)
    axarr.plot([i for i in range(0,100,20)], SVG, label="SVG")
    axarr.plot([i for i in range(0,100,20)], SVG_TE, label="SVG_TE")
    axarr.plot([i for i in range(0,100,20)], SVTG, label="SVTG")
    axarr.plot([i for i in range(0,100,20)], SPOTS_STP, label="SPOTS_STP")

    a=axarr.get_xticks().tolist()
    a[1]='100'
    a[0]=''
    a[2]=''
    a[3]='80'
    a[4]=''
    a[5]='60'
    a[6] = ''
    a[7]='40'
    a[8] = ''
    a[9]='20'
    axarr.set_xticklabels(a)
    axarr.set_xlabel("Dataset %", fontsize=15)
    axarr.set_ylabel("MAE", fontsize=15)
    axarr.legend()
    plt.show()

if __name__ == '__main__':
    # plot_training_scores()
    # plot_model_performance()
    # send_email()
    # find_marker()
    # plot_learning_rate()
    # plot_edge_case_qualitative()
    # plot_edge_case_t4_qualitative()
    # plot_edge_case_scene_synthesis_qualitative()
    plot_edge_case_reduced_data_qualitative()
    # plot_reduced_data()


