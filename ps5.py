"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np
import os

from ps5_utils import run_kalman_filter, run_particle_filter
input_dir = "input"
output_dir = "output"
np.random.seed(1)
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """

        # state vector X
        self.state = np.array([init_x, init_y, 0., 0.])

        # covariance matrix Σt
        self.Σ = np.array(
                    [[0.1, 0., 0., 0.],
                    [0., 0.1, 0., 0.],
                    [0., 0., 0.1, 0.],
                    [0., 0., 0., 0.1]])

        # state transition matrix Dt
        self.Dt = np.array([[1.,0.,1.,0.],
                           [0.,1.,0.,1.],
                           [0.,0.,1.,0.],
                           [0.,0.,0.,1.]])

        # measurement matrix Mt
        self.Mt = np.array( [[1., 0., 0., 0.],
                            [0., 1., 0., 0.]])


        #Identity matris I
        self.I = np.eye(4)

        #Q: process noise matrix
        self.Q = 0.15 * np.eye(4)

        #R: measurement noise matrix
        self.R = 0.15 * np.eye(2)

    def predict(self):
        self.state = np.matmul(self.state,  self.Dt)
        self.Σ = np.matmul(np.matmul(self.Dt,self.Σ), self.Dt.transpose()) + self.Q

    def correct(self, meas_x, meas_y):
        Yt = np.array([meas_x,meas_y])

        #Kalman gain calculation
        self.Kalman_gain = self.Σ @ self.Mt.transpose() @ np.linalg.inv(self.Mt @ self.Σ @ self.Mt.transpose() + self.R)


        self.state = self.state + np.matmul(self.Kalman_gain,(Yt - np.matmul(self.Mt , self.state)))

        self.Σ = np.matmul((self.I - np.matmul(self.Kalman_gain,self.Mt)),self.Σ)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template,alpha = 0.05, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """

        self.frame = frame.copy()
        self.frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        self.weights = np.ones(self.num_particles,dtype=float)/self.num_particles
        self.alpha = alpha
        self.template_resize = kwargs.get('template_resize')
        # self.template_resize = True
        self.step = 0

        x = int(self.template_rect['x'])
        y = int(self.template_rect['y'])
        w = int(self.template_rect['w'])
        h = int(self.template_rect['h'])

        x_template, y_template = int(x + w / 2), int(y + h / 2)

        self.x_template = x_template
        self.y_template = y_template
        self.window_w = w
        self.window_h = h
        self.mean_x = 0
        self.mean_y = 0

        # initiallize particles uniformly over template window
        x_y_coor  = [[x, y] for x, y in zip(
            np.random.uniform(self.x_template - self.window_w / 2, self.x_template + self.window_w / 2, self.num_particles),
            np.random.uniform(self.y_template - self.window_w / 2, self.y_template + self.window_h / 2, self.num_particles))]

        # if windows size need to changes:
        if self.template_resize:
            template_sizes = [[w, h] for w, h in zip(
                np.random.uniform(self.window_w - self.sigma_dyn, self.window_w + self.sigma_dyn, self.num_particles),
                np.random.uniform(self.window_h - self.sigma_dyn, self.window_h + self.sigma_dyn, self.num_particles))
                # np.random.uniform(self.window_w , self.window_w , self.num_particles),
                # np.random.uniform(self.window_h , self.window_h , self.num_particles))
                              ]
            self.particles = np.array([[coordinate[0], coordinate[1], template_size[0], template_size[1]] for coordinate, template_size
                                       in zip(x_y_coor, template_sizes)])
        else:
            self.particles = np.array(x_y_coor)


    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        if (frame_cutout.shape[0] <= 0 or frame_cutout.shape[1] <= 0):
            return 0
        template = cv2.resize(template, (np.shape(frame_cutout)[1],np.shape(frame_cutout)[0]))
        return np.exp(-np.mean(np.square(np.subtract(template, frame_cutout))) / (2 * (self.sigma_exp ** 2)))

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        self.weights = np.where(self.weights == np.NaN,0,self.weights)
        resampled_particles_indices = np.random.choice(self.num_particles, size=self.num_particles, replace=True,
                                                       p=self.weights)
        self.particles = self.particles[resampled_particles_indices]
        self.weights = self.weights[resampled_particles_indices]
        self.weights = self.weights / np.sum(self.weights)
        return self.particles

    def template_cutout(self, x, y, w, h):
        """
        :param x: x coordinate of window center
        :param y: y coordinate of window center
        :param w: window width
        :param h: window height
        :return: gray template cut out [y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        """

        x_top_left, x_bottom_right = max(0, x - w // 2), min(self.width, x + w // 2)
        y_top_left, y_bottom_right = max(0, y - h // 2), min(self.height, y + h // 2)
        return self.frame_gray[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        self.frame = frame.copy()
        self.frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # Apply dynamic model
        if self.template_resize:
            self.particles += np.random.normal(0, self.sigma_dyn, self.num_particles * 4).reshape(self.particles.shape)
        else:
            self.particles += np.random.normal(0, self.sigma_dyn, self.num_particles * 2).reshape(self.particles.shape)

        # find template window for each particle
        if self.template_resize:
            particle_windows = [self.template_cutout(p[0], p[1], p[2], p[3])
                             for p in self.particles.astype(int)]
        else:
            particle_windows = [self.template_cutout(p[0], p[1], self.window_w, self.window_h)
                             for p in self.particles.astype(int)]

        # get weight for each particle
        self.weights = [self.get_error_metric(self.template, particle_window) for particle_window in particle_windows]
        self.weights = self.weights/np.sum(self.weights)

        self.resample_particles()

        self.mean_x = np.average(self.particles[:, 0], weights=self.weights)
        self.mean_y = np.average(self.particles[:, 1], weights=self.weights)
        if self.template_resize:
            self.window_w = np.average(self.particles[:, 2], weights=self.weights)
            self.window_h = np.average(self.particles[:, 3], weights=self.weights)
        x_left = max(0, int(self.mean_x - self.window_w / 2))
        x_right = min(self.width, int(self.mean_x + self.window_w / 2 + 1))
        y_left = max(0, int(self.mean_y - self.window_h / 2))
        y_right = min(self.height, int(self.mean_y + self.window_h / 2 + 1))

        # best template at current
        self.best = np.array(self.frame_gray[y_left:y_right, x_left:x_right])

        # combine template and best for IIR method
        if self.alpha != 0:
            self.best = cv2.resize(self.best, (self.template.shape[1], self.template.shape[0]))
            self.template = np.add(self.alpha * self.best, (1 - self.alpha) * self.template)
        # print()
        # self.step +=1
        # u_weighted_mean = 0
        # v_weighted_mean = 0
        #
        # for i in range(self.num_particles):
        #     u_weighted_mean += self.particles[i, 0] * self.weights[i]
        #     v_weighted_mean += self.particles[i, 1] * self.weights[i]
        # print('step',self.step,'sum weight' , np.sum(self.weights), 'u_weighted_mean',u_weighted_mean, 'v_weighted_mean ', v_weighted_mean,  )

        return None


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """


        for pos, particle in enumerate(self.particles):
            frame_in = cv2.circle(frame_in, (int(particle[0]),int(particle[1])), radius=1, color=(0, 255, 255), thickness=1)
        frame_in = cv2.rectangle(frame_in,
                                 tuple((int(self.mean_x - self.window_w / 2), int(self.mean_y - self.window_h / 2))),
                                 tuple((int(self.mean_x + self.window_w / 2), int(self.mean_y + self.window_h / 2))),
                                 color = (0, 255, 255), thickness=1)

        radius = 0
        for pos, particle in enumerate(self.particles):
            radius += ((self.mean_x - particle[0]) ** 2 + (self.mean_y - particle[1]) ** 2) ** (0.5) * self.weights[pos]

        frame_in = cv2.circle(frame_in, (int(self.mean_x), int(self.mean_y)), radius = int(radius), color=(0, 0, 255), thickness=2)
        return None

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)



def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 3.  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha = alpha,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 1.7   # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 2.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.01  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha = alpha,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 2000  # Define the number of particles
    sigma_mse =  1.5   # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20.  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.05  # Set a value for alpha
    ###############################################
    # TO PRODUCE IMAGE IN THE REPORT, USE SEED BELOW
    ###############################################
    # np.random.seed(100)
    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 600  # Define the number of particles
    sigma_md = 1.7  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 1.7  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0
    ###############################################
    # TO PRODUCE IMAGE IN THE REPORT, USE SEED BELOW
    ###############################################
    np.random.seed(101)
    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect,
        alpha=alpha,
        template_resize = True)  # Add more if you need to
    return out
    # num_particles = 800  # Define the number of particles
    # sigma_md = 2.2  # Define the value of sigma for the measurement exponential equation
    # sigma_dyn = 2.  # Define the value of sigma for the particles movement (dynamics)
    # alpha = 0
    # np.random.seed(100)

def part_5(obj_class, template_rect, save_frames, input_folder,number_of_objects,starting_frame):
    # part 5 is consolidated in experiment.py
    return None


def part_6(obj_class, template_rect, save_frames, input_folder):
    num_particles = 800  # Define the number of particles
    sigma_md = 2.5# Define the value of sigma for the measurement exponential equation
    sigma_dyn = 2.5  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.18
    ###############################################
    # TO PRODUCE IMAGE IN THE REPORT, USE SEED BELOW
    ###############################################
    # np.random.seed(1000)
    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect,
        alpha=alpha,
        template_resize = True)  # Add more if you need to
    return out
