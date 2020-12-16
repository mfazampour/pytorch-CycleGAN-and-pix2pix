import numpy as np
import torch
def get_index(lmark_truth, lmark_pred, number):
    index_t = torch.where(lmark_truth == number)
    index_p = torch.where(lmark_pred == number)
#
    if len(index_p[0])<1:
        if torch.mean(index_t[0].float()) > 44:
            ind_x = 95
        else:
            ind_x = - 10
        if torch.mean(index_t[1].float()) > 33:
            ind_y = 75
        else:
            ind_y = - 10
        if torch.mean(index_t[2].float()) > 39:
            ind_z = 89
        else:
            ind_z = - 10
        index_p = [torch.tensor([ind_x]),torch.tensor([ind_y]),torch.tensor([ind_z])]
    return index_t , index_p
#

def get_distance_lmark(lmark_truth, lmark_pred, device):
    landmark_tot_distance = []
    #print(lmark_pred.shape)
    #print(lmark_truth.shape)

    for i in range (len(lmark_truth)):

        for landmark in range(int(torch.max(lmark_truth))):
            diff = 0
            index_t , index_p = get_index(lmark_truth[i,:,:,:],lmark_pred[i,:,:,:], landmark+1)
            #index_p = get_index(lmark_pred[i,:,:,:], landmark+1)
           # print(f" index_p{index_p}    index_t {index_t}")


            meanx_t = torch.mean(index_t[0].float())
            meany_t = torch.mean(index_t[1].float())
            meanz_t = torch.mean(index_t[2].float())

            meanx_p = torch.mean(index_p[0].float())
            meany_p = torch.mean(index_p[1].float())
            meanz_p = torch.mean(index_p[2].float())
          #  print(f" mean_xp: {meanx_p} mean_xt: {meanx_t}")
          #  print(f" mean_yp: {meany_p} mean_yt: {meany_t}")
          #  print(f" mean_zp: {meanz_p} mean_zt: {meanz_t}")
            diff += torch.abs(meanx_t - meanx_p)
            diff += torch.abs(meany_t - meany_p)
            diff += torch.abs(meanz_t - meanz_p)
            diff = diff /3

          #  if diff > 20.0 :
            #    print(f'   p_X {index_p[0]}   mean_xp: {meanx_p}  \n  t_X {index_t[0]}   mean_xt: {meanx_t}   \n difference {(meanx_t - meanx_p)} abs difference {torch.abs(meanx_t - meanx_p)} ')



            landmark_tot_distance.append(torch.tensor(diff).to(device))
    #landmark_tot_distance = torch.tensor(landmark_tot_distance).to(device)
    return torch.mean(torch.stack(landmark_tot_distance))


