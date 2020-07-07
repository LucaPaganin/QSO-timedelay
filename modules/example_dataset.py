from . import data_generator as dg

t_max = 4000
n_samples = 100
delay = 100
seed = 0
image_ratio = 1
add_noise = True
n_gaps = 5
gap_size = 3
relative_err = 0.02

datagen = dg.DataGenerator(t_max=t_max,
                           sampling_rate=n_samples,
                           delay=delay,
                           seed=seed,
                           image_ratio=image_ratio,
                           add_noise=add_noise,
                           n_gaps=n_gaps,
                           gap_size=gap_size,
                           relative_err=relative_err)

means, sigmas, heights = datagen.generate_gaussian_signal_parameters()

dataset = datagen.generate_single_realization_dataset(means=means,
                                                      sigmas=sigmas,
                                                      heights=heights,
                                                      gap_size=gap_size)

dataset.to_csv("example_output.csv")