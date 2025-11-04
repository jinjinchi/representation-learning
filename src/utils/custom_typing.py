from typing import NamedTuple, Tuple
import torch


class GanLossOutput(NamedTuple):
    discriminator: torch.Tensor
    generator: torch.Tensor


class EncoderOutput(NamedTuple):
    representation: torch.Tensor
    feature: torch.Tensor


class ColoredMNISTData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    fg_label: torch.Tensor
    bg_label: torch.Tensor
    digit_label: torch.Tensor
    index: torch.Tensor


class threed_teacher_Data(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    floor_color_labels_x: torch.Tensor
    wall_color_labels_x: torch.Tensor
    object_color_labels_x: torch.Tensor
    object_scale_labels: torch.Tensor
    object_shape_labels: torch.Tensor
    scene_orientation_labels: torch.Tensor
    index: torch.Tensor


class SDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    global_M_R_x: torch.Tensor
    global_M_R_x_prime: torch.Tensor
    global_M_R_y: torch.Tensor
    global_M_R_y_prime: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor
    global_gradient_penalty_x: torch.Tensor
    global_gradient_penalty_y: torch.Tensor
    local_gradient_penalty_x: torch.Tensor
    local_gradient_penalty_y: torch.Tensor
    weight_x_joint: torch.Tensor
    weight_y_joint: torch.Tensor
    weight_x_margin: torch.Tensor
    weight_y_margin: torch.Tensor
    card_logits: torch.Tensor
    suit_logits: torch.Tensor
    input_x: torch.Tensor
    input_x_prime: torch.Tensor
    input_y: torch.Tensor
    input_y_prime: torch.Tensor
    

class WeightNetOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    weight_x_joint: torch.Tensor
    weight_y_joint: torch.Tensor
    weight_x_margin: torch.Tensor
    weight_y_margin: torch.Tensor


class EDIMOutputs(NamedTuple):
    global_mutual_M_R_x: torch.Tensor
    global_mutual_M_R_x_prime: torch.Tensor
    global_mutual_M_R_y: torch.Tensor
    global_mutual_M_R_y_prime: torch.Tensor
    local_mutual_M_R_x: torch.Tensor
    local_mutual_M_R_x_prime: torch.Tensor
    local_mutual_M_R_y: torch.Tensor
    local_mutual_M_R_y_prime: torch.Tensor
    global_gradient_penalty_x: torch.Tensor
    global_gradient_penalty_y: torch.Tensor
    local_gradient_penalty_x: torch.Tensor
    local_gradient_penalty_y: torch.Tensor
    weight_x_joint: torch.Tensor
    weight_y_joint: torch.Tensor
    weight_x_margin: torch.Tensor
    weight_y_margin: torch.Tensor
    input_x: torch.Tensor
    input_x_prime: torch.Tensor
    input_y: torch.Tensor
    input_y_prime: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor
    representation_x_joint: torch.Tensor
    representation_y_joint: torch.Tensor
    representation_x_margin: torch.Tensor
    representation_y_margin: torch.Tensor
    critic_x: torch.Tensor
    critic_y: torch.Tensor
    critic_x_prime: torch.Tensor
    critic_y_prime: torch.Tensor
    disen_weight_x_joint: torch.Tensor
    disen_weight_y_joint: torch.Tensor
    disen_weight_x_margin: torch.Tensor
    disen_weight_y_margin: torch.Tensor


class SDIMLosses(NamedTuple):
    total_loss: torch.Tensor
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    shared_loss: torch.Tensor
    card_classif_loss: torch.Tensor
    suit_classif_loss: torch.Tensor
    card_accuracy: torch.Tensor
    suit_accuracy: torch.Tensor



class GenLosses(NamedTuple):
    encoder_loss: torch.Tensor
    local_mutual_loss: torch.Tensor
    global_mutual_loss: torch.Tensor
    loss_disen_g: torch.Tensor


class ClassifLosses(NamedTuple):
    classif_loss: torch.Tensor
    card_classif_loss: torch.Tensor
    suit_classif_loss: torch.Tensor
    card_accuracy: torch.Tensor
    suit_accuracy: torch.Tensor


class DiscrLosses(NamedTuple):
    gan_loss_d: torch.Tensor


class DiscrLosses(NamedTuple):
    loss_dis_d: torch.Tensor
    loss_dis: torch.Tensor
    critic_gp: torch.Tensor


class GeneratorOutputs(NamedTuple):
    real_x: torch.Tensor
    fake_x: torch.Tensor
    real_y: torch.Tensor
    fake_y: torch.Tensor
    exclusive_x: torch.Tensor
    exclusive_y: torch.Tensor


class DiscriminatorOutputs(NamedTuple):
    disentangling_information_x: torch.Tensor
    disentangling_information_x_prime: torch.Tensor
    disentangling_information_y: torch.Tensor
    disentangling_information_y_prime: torch.Tensor


class ClassifierOutputs(NamedTuple):
    card_logits: torch.Tensor
    suit_logits: torch.Tensor


class SDIMClassifyLosses(NamedTuple):
    classify_loss: torch.Tensor
    digit_classif_loss: torch.Tensor
    color_bg_classif_loss: torch.Tensor
    color_fg_classif_loss: torch.Tensor
    digit_accuracy: torch.Tensor
    color_bg_accuracy: torch.Tensor
    color_fg_accuracy: torch.Tensor


class TrainOutputsMnist(NamedTuple):
    digit_logits: torch.Tensor
    color_bg_logits: torch.Tensor
    color_fg_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class TrainClassifyOutputs3DShapes(NamedTuple):
    floor_color_logits: torch.Tensor
    wall_color_logits: torch.Tensor
    object_color_logits: torch.Tensor
    object_scale_logits: torch.Tensor
    object_shape_logits: torch.Tensor
    scene_orientation_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class TrainLossesMnist(NamedTuple):
    classify_loss: torch.Tensor
    digit_classif_loss: torch.Tensor
    color_bg_classif_loss: torch.Tensor
    color_fg_classif_loss: torch.Tensor
    digit_accuracy: torch.Tensor
    color_bg_accuracy: torch.Tensor
    color_fg_accuracy: torch.Tensor


class TrainOutputsIAM(NamedTuple):
    writer_logits: torch.Tensor
    word_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class TrainOutputsCar3d(NamedTuple):
    elevation_logits: torch.Tensor
    azimuth_logits: torch.Tensor
    object_type_logits: torch.Tensor
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class TrainOutputsSplitCelebA(NamedTuple):
    attributes_logits: dict
    shared_x: torch.Tensor
    shared_y: torch.Tensor


class TrainLosses3DShapes(NamedTuple):
    classify_loss: torch.Tensor
    floor_color_classif_loss: torch.Tensor
    wall_color_classif_loss: torch.Tensor
    object_color_classif_loss: torch.Tensor
    object_scale_classif_loss: torch.Tensor
    object_shape_classif_loss: torch.Tensor
    scene_orientation_classif_loss: torch.Tensor
    floor_color_accuracy: torch.Tensor
    wall_color_accuracy: torch.Tensor
    object_color_accuracy: torch.Tensor
    object_scale_accuracy: torch.Tensor
    object_shape_accuracy: torch.Tensor
    scene_orientation_accuracy: torch.Tensor


class TrainLossesIAM(NamedTuple):
    classify_loss: torch.Tensor
    writer_classif_loss: torch.Tensor
    word1_classif_loss: torch.Tensor
    # word2_classif_loss: torch.Tensor
    writer_accuracy: torch.Tensor
    word1_accuracy: torch.Tensor
    # word2_accuracy: torch.Tensor


class TrainLossesCar3d(NamedTuple):
    classify_loss: torch.Tensor
    elevation_classif_loss: torch.Tensor
    azimuth_classif_loss: torch.Tensor
    object_type_classif_loss: torch.Tensor
    elevation_accuracy: torch.Tensor
    azimuth_accuracy: torch.Tensor
    object_type_accuracy: torch.Tensor


class TrainLossesCelebA(NamedTuple):
    classif_loss: torch.Tensor
    attr_classif_loss: dict
    attr_accuracy: dict
    attr_precisions: dict
    attr_recalls: dict
    attr_f1_scores: dict


class ThreeDShapesData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    floor_color_labels_y: torch.Tensor
    wall_color_labels_y: torch.Tensor
    object_color_labels_y: torch.Tensor
    object_scale_labels: torch.Tensor
    object_shape_labels: torch.Tensor
    scene_orientation_labels: torch.Tensor
    index: torch.Tensor


class CardData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    card_x: torch.Tensor
    card_y: torch.Tensor
    suit_x: torch.Tensor
    suit_y: torch.Tensor


class Car3dData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    elevation_labels_x: torch.Tensor
    azimuth_labels_x: torch.Tensor
    elevation_labels_y: torch.Tensor
    azimuth_labels_y: torch.Tensor
    object_type_labels: torch.Tensor


class SplitCelebAData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    attributes: dict
    

class KDEFData(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    person_labels: torch.Tensor
    glasses_labels_x: torch.Tensor
    glasses_labels_y: torch.Tensor
