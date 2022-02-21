real_target = 1
fake_target = 0


def least_squares_generator_loss(j):
    return 0.5 * ((j - real_target) ** 2).mean()


def squared_gan_loss(value, target):
    return ((value - target) ** 2).mean()

def least_squares_disc_loss(
        r_j,
        f_j,
        real_target=real_target,
        fake_target=fake_target):

    return 0.5 * (((r_j - real_target) ** 2).mean() + ((f_j - fake_target) ** 2).mean())
