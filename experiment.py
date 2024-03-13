"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import os

import cv2
import numpy as np

import ps5

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-b-1.png'),
        28: os.path.join(output_dir, 'ps5-1-b-2.png'),
        57: os.path.join(output_dir, 'ps5-1-b-3.png'),
        97: os.path.join(output_dir, 'ps5-1-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1b(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "circle"))


def part_1c():
    print("Part 1c")

    template_loc = {'x': 311, 'y': 217}
    save_frames = {
        12: os.path.join(output_dir, 'ps5-1-c-1.png'),
        30: os.path.join(output_dir, 'ps5-1-c-2.png'),
        81: os.path.join(output_dir, 'ps5-1-c-3.png'),
        155: os.path.join(output_dir, 'ps5-1-c-4.png')
    }

    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_1c(ps5.KalmanFilter, template_loc, save_frames,
                os.path.join(input_dir, "walking"))


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {
        8: os.path.join(output_dir, 'ps5-2-a-1.png'),
        28: os.path.join(output_dir, 'ps5-2-a-2.png'),
        57: os.path.join(output_dir, 'ps5-2-a-3.png'),
        97: os.path.join(output_dir, 'ps5-2-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2a(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "circle"))


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    save_frames = {
        12: os.path.join(output_dir, 'ps5-2-b-1.png'),
        28: os.path.join(output_dir, 'ps5-2-b-2.png'),
        57: os.path.join(output_dir, 'ps5-2-b-3.png'),
        97: os.path.join(output_dir, 'ps5-2-b-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_2b(
        ps5.ParticleFilter,  # particle filter model class
        template_loc,
        save_frames,
        os.path.join(input_dir, "pres_debate_noisy"))


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    save_frames = {
        20: os.path.join(output_dir, 'ps5-3-a-1.png'),
        48: os.path.join(output_dir, 'ps5-3-a-2.png'),
        158: os.path.join(output_dir, 'ps5-3-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_3(
        ps5.AppearanceModelPF,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pres_debate"))


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}

    save_frames = {
        40: os.path.join(output_dir, 'ps5-4-a-1.png'),
        100: os.path.join(output_dir, 'ps5-4-a-2.png'),
        240: os.path.join(output_dir, 'ps5-4-a-3.png'),
        300: os.path.join(output_dir, 'ps5-4-a-4.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_4(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "pedestrians"))


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    number_of_objects = 3
    template_rect  = [{'x': 76, 'y': 148, 'w': 70, 'h': 180},{'x': 300, 'y': 200, 'w': 38, 'h': 116},{'x': 0, 'y': 175, 'w': 55, 'h': 170}]
    starting_frame = [0,0,27]

    save_frames = {
        28: os.path.join(output_dir, 'ps5-5-a-1.png'),
        55: os.path.join(output_dir, 'ps5-5-a-2.png'),
        70: os.path.join(output_dir, 'ps5-5-a-3.png')
    }
    num_particles = 500  # Define the number of particles
    sigma_md = 3.5  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 25.5  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.0005
    np.random.seed(100)

    imgs_list = [f for f in os.listdir(os.path.join(input_dir, "TUD-Campus"))
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = [None,None,None]
    pf = [None,None,None]
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(os.path.join(input_dir, "TUD-Campus"), img))

        for i in range(number_of_objects):
            if frame_num >= starting_frame[i]:
                # Extract template and initialize (one-time only)
                if template[i] is None:
                    template[i] = frame[int(template_rect[i]['y']):
                                     int(template_rect[i]['y'] + template_rect[i]['h']),
                               int(template_rect[i]['x']):
                               int(template_rect[i]['x'] + template_rect[i]['w'])]

                    pf[i] = ps5.MDParticleFilter(frame, template[i],
                                             template_coords = template_rect[i],
                                             num_particles=num_particles,
                                             sigma_exp=sigma_md,
                                             sigma_dyn=sigma_dyn,
                                             alpha=alpha,
                                             template_resize = False )

                # Process frame
                pf[i].process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            for i in range(number_of_objects):
                if pf[i] != None:
                    pf[i].render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            cv2.imwrite(save_frames[frame_num], out_frame)

        # Update frame number
        frame_num += 1
        print('Working on frame {}'.format(frame_num))
    return 0

def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_rect = {'x': 77, 'y': 32, 'w': 65, 'h': 125}

    save_frames = {
        59: os.path.join(output_dir, 'ps5-6-a-1.png'),
        159: os.path.join(output_dir, 'ps5-6-a-2.png'),
        185: os.path.join(output_dir, 'ps5-6-a-3.png')
    }
    # Define process and measurement arrays if you want to use other than the
    # default.
    ps5.part_6(
        ps5.MDParticleFilter,  # particle filter model class
        template_rect,
        save_frames,
        os.path.join(input_dir, "follow"))


if __name__ == '__main__':
    np.random.seed(5)
    # REMOVE THE SEED ABOVE, USE SEED IN PS5.PY AND RUN EACH PART SEPARATELY TO REPRODUCE THE REPORT IMAGES
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    part_4()
    # part_5()
    # part_6()
