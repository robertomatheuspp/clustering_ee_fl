

def energy_cost_comm(number_of_trainable_parameters):
    """
    Here we are calculating the energy needed to send once the model to the server.
    Particularly, we will consider the modeling and parameters of https://arxiv.org/pdf/2209.07124.pdf

    We will assume that each parameter is store as a float32, meaning 4 bytes

    :param number_of_trainable_parameters:
    :return: necessary energy to transmit this from uplink and donwlink
    """

    # We will start by calculating how many nano seconds we need to send the data considering the exact same conditions
    # as in the reference paper.

    power_bs = 20
    power_client = 9

    length_data = number_of_trainable_parameters*4

    t_phy = 20
    lsf = 16
    lrts = 160
    lcts = 112
    lack = 240
    sigma_leg = 4
    Ls = 24 #size ofdm symbo

    time_rst = t_phy + (lsf + lrts)*sigma_leg/Ls
    time_cts = t_phy + (lsf + lcts) * sigma_leg / Ls
    time_ack = t_phy + (lsf + lack) * sigma_leg / Ls

    time_sifs = 16
    time_difs = 34
    time_empty = 9

    the_su = 100
    lmac = 320


    time_data = the_su + (lsf + lmac + length_data)*sigma_leg/Ls

    total_time_nano_seconds = (time_rst + time_sifs + time_difs + time_cts + time_ack + time_empty + time_data)

    total_time = total_time_nano_seconds*1e-9 # seconds

    return total_time#, total_time*power_bs, total_time*power_client

