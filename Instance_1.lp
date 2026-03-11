\ Model Instance_1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  38 take[0] + 28 take[1] + 72 take[2] + 95 take[3] + 50 take[4]
   + 18 take[5] + 110 take[6] + 62 take[7] + 30 take[8] + 36 take[9]
Subject To
 weight_limit: 2.2 take[0] + 1.8 take[1] + 3.5 take[2] + 4.8 take[3]
   + 2.6 take[4] + 1.5 take[5] + 5.5 take[6] + 3 take[7] + 1.9 take[8]
   + 2.1 take[9] <= 18
 size_sum_limit: L + W + H <= 158
 sym1: L - W >= 0
 sym2: W - H >= 0
 one_orientation_0: - take[0] + orient_0_0 + orient_0_1 + orient_0_2
   + orient_0_3 + orient_0_4 + orient_0_5 = 0
 link_lx_0: lx[0] - 18 orient_0_0 - 20 orient_0_1 - 30 orient_0_2
   - 30 orient_0_3 - 18 orient_0_4 - 20 orient_0_5 = 0
 link_ly_0: ly[0] - 20 orient_0_0 - 18 orient_0_1 - 20 orient_0_2
   - 18 orient_0_3 - 30 orient_0_4 - 30 orient_0_5 = 0
 link_lz_0: lz[0] - 30 orient_0_0 - 30 orient_0_1 - 18 orient_0_2
   - 20 orient_0_3 - 20 orient_0_4 - 18 orient_0_5 = 0
 fit_x_0: - L + 200 take[0] + x[0] + lx[0] <= 200
 fit_y_0: - W + 200 take[0] + y[0] + ly[0] <= 200
 fit_z_0: - H + 200 take[0] + z[0] + lz[0] <= 200
 activate_x_0: - 200 take[0] + x[0] <= 0
 activate_y_0: - 200 take[0] + y[0] <= 0
 activate_z_0: - 200 take[0] + z[0] <= 0
 one_orientation_1: - take[1] + orient_1_0 + orient_1_1 + orient_1_2
   + orient_1_3 + orient_1_4 + orient_1_5 = 0
 link_lx_1: lx[1] - 25 orient_1_0 - 25 orient_1_1 - 16 orient_1_2
   - 15 orient_1_3 - 15 orient_1_4 - 16 orient_1_5 = 0
 link_ly_1: ly[1] - 16 orient_1_0 - 15 orient_1_1 - 15 orient_1_2
   - 25 orient_1_3 - 16 orient_1_4 - 25 orient_1_5 = 0
 link_lz_1: lz[1] - 15 orient_1_0 - 16 orient_1_1 - 25 orient_1_2
   - 16 orient_1_3 - 25 orient_1_4 - 15 orient_1_5 = 0
 fit_x_1: - L + 200 take[1] + x[1] + lx[1] <= 200
 fit_y_1: - W + 200 take[1] + y[1] + ly[1] <= 200
 fit_z_1: - H + 200 take[1] + z[1] + lz[1] <= 200
 activate_x_1: - 200 take[1] + x[1] <= 0
 activate_y_1: - 200 take[1] + y[1] <= 0
 activate_z_1: - 200 take[1] + z[1] <= 0
 one_orientation_2: - take[2] + orient_2_0 + orient_2_1 + orient_2_2
   + orient_2_3 + orient_2_4 + orient_2_5 = 0
 link_lx_2: lx[2] - 30 orient_2_0 - 30 orient_2_1 - 40 orient_2_2
   - 22 orient_2_3 - 40 orient_2_4 - 22 orient_2_5 = 0
 link_ly_2: ly[2] - 40 orient_2_0 - 22 orient_2_1 - 22 orient_2_2
   - 30 orient_2_3 - 30 orient_2_4 - 40 orient_2_5 = 0
 link_lz_2: lz[2] - 22 orient_2_0 - 40 orient_2_1 - 30 orient_2_2
   - 40 orient_2_3 - 22 orient_2_4 - 30 orient_2_5 = 0
 fit_x_2: - L + 200 take[2] + x[2] + lx[2] <= 200
 fit_y_2: - W + 200 take[2] + y[2] + ly[2] <= 200
 fit_z_2: - H + 200 take[2] + z[2] + lz[2] <= 200
 activate_x_2: - 200 take[2] + x[2] <= 0
 activate_y_2: - 200 take[2] + y[2] <= 0
 activate_z_2: - 200 take[2] + z[2] <= 0
 one_orientation_3: - take[3] + orient_3_0 + orient_3_1 + orient_3_2
   + orient_3_3 + orient_3_4 + orient_3_5 = 0
 link_lx_3: lx[3] - 35 orient_3_0 - 35 orient_3_1 - 55 orient_3_2
   - 55 orient_3_3 - 20 orient_3_4 - 20 orient_3_5 = 0
 link_ly_3: ly[3] - 55 orient_3_0 - 20 orient_3_1 - 35 orient_3_2
   - 20 orient_3_3 - 35 orient_3_4 - 55 orient_3_5 = 0
 link_lz_3: lz[3] - 20 orient_3_0 - 55 orient_3_1 - 20 orient_3_2
   - 35 orient_3_3 - 55 orient_3_4 - 35 orient_3_5 = 0
 fit_x_3: - L + 200 take[3] + x[3] + lx[3] <= 200
 fit_y_3: - W + 200 take[3] + y[3] + ly[3] <= 200
 fit_z_3: - H + 200 take[3] + z[3] + lz[3] <= 200
 activate_x_3: - 200 take[3] + x[3] <= 0
 activate_y_3: - 200 take[3] + y[3] <= 0
 activate_z_3: - 200 take[3] + z[3] <= 0
 one_orientation_4: - take[4] + orient_4_0 + orient_4_1 + orient_4_2
   + orient_4_3 + orient_4_4 + orient_4_5 = 0
 link_lx_4: lx[4] - 20 orient_4_0 - 20 orient_4_1 - 25 orient_4_2
   - 35 orient_4_3 - 35 orient_4_4 - 25 orient_4_5 = 0
 link_ly_4: ly[4] - 25 orient_4_0 - 35 orient_4_1 - 20 orient_4_2
   - 20 orient_4_3 - 25 orient_4_4 - 35 orient_4_5 = 0
 link_lz_4: lz[4] - 35 orient_4_0 - 25 orient_4_1 - 35 orient_4_2
   - 25 orient_4_3 - 20 orient_4_4 - 20 orient_4_5 = 0
 fit_x_4: - L + 200 take[4] + x[4] + lx[4] <= 200
 fit_y_4: - W + 200 take[4] + y[4] + ly[4] <= 200
 fit_z_4: - H + 200 take[4] + z[4] + lz[4] <= 200
 activate_x_4: - 200 take[4] + x[4] <= 0
 activate_y_4: - 200 take[4] + y[4] <= 0
 activate_z_4: - 200 take[4] + z[4] <= 0
 one_orientation_5: - take[5] + orient_5_0 + orient_5_1 + orient_5_2
   + orient_5_3 + orient_5_4 + orient_5_5 = 0
 link_lx_5: lx[5] - 15 orient_5_0 - 15 orient_5_1 - 18 orient_5_2
   - 18 orient_5_3 - 12 orient_5_4 - 12 orient_5_5 = 0
 link_ly_5: ly[5] - 18 orient_5_0 - 12 orient_5_1 - 12 orient_5_2
   - 15 orient_5_3 - 15 orient_5_4 - 18 orient_5_5 = 0
 link_lz_5: lz[5] - 12 orient_5_0 - 18 orient_5_1 - 15 orient_5_2
   - 12 orient_5_3 - 18 orient_5_4 - 15 orient_5_5 = 0
 fit_x_5: - L + 200 take[5] + x[5] + lx[5] <= 200
 fit_y_5: - W + 200 take[5] + y[5] + ly[5] <= 200
 fit_z_5: - H + 200 take[5] + z[5] + lz[5] <= 200
 activate_x_5: - 200 take[5] + x[5] <= 0
 activate_y_5: - 200 take[5] + y[5] <= 0
 activate_z_5: - 200 take[5] + z[5] <= 0
 one_orientation_6: - take[6] + orient_6_0 + orient_6_1 + orient_6_2
   + orient_6_3 + orient_6_4 + orient_6_5 = 0
 link_lx_6: lx[6] - 40 orient_6_0 - 40 orient_6_1 - 60 orient_6_2
   - 60 orient_6_3 - 22 orient_6_4 - 22 orient_6_5 = 0
 link_ly_6: ly[6] - 60 orient_6_0 - 22 orient_6_1 - 22 orient_6_2
   - 40 orient_6_3 - 40 orient_6_4 - 60 orient_6_5 = 0
 link_lz_6: lz[6] - 22 orient_6_0 - 60 orient_6_1 - 40 orient_6_2
   - 22 orient_6_3 - 60 orient_6_4 - 40 orient_6_5 = 0
 fit_x_6: - L + 200 take[6] + x[6] + lx[6] <= 200
 fit_y_6: - W + 200 take[6] + y[6] + ly[6] <= 200
 fit_z_6: - H + 200 take[6] + z[6] + lz[6] <= 200
 activate_x_6: - 200 take[6] + x[6] <= 0
 activate_y_6: - 200 take[6] + y[6] <= 0
 activate_z_6: - 200 take[6] + z[6] <= 0
 one_orientation_7: - take[7] + orient_7_0 + orient_7_1 + orient_7_2
   + orient_7_3 + orient_7_4 + orient_7_5 = 0
 link_lx_7: lx[7] - 45 orient_7_0 - 25 orient_7_1 - 20 orient_7_2
   - 45 orient_7_3 - 20 orient_7_4 - 25 orient_7_5 = 0
 link_ly_7: ly[7] - 25 orient_7_0 - 20 orient_7_1 - 25 orient_7_2
   - 20 orient_7_3 - 45 orient_7_4 - 45 orient_7_5 = 0
 link_lz_7: lz[7] - 20 orient_7_0 - 45 orient_7_1 - 45 orient_7_2
   - 25 orient_7_3 - 25 orient_7_4 - 20 orient_7_5 = 0
 fit_x_7: - L + 200 take[7] + x[7] + lx[7] <= 200
 fit_y_7: - W + 200 take[7] + y[7] + ly[7] <= 200
 fit_z_7: - H + 200 take[7] + z[7] + lz[7] <= 200
 activate_x_7: - 200 take[7] + x[7] <= 0
 activate_y_7: - 200 take[7] + y[7] <= 0
 activate_z_7: - 200 take[7] + z[7] <= 0
 one_orientation_8: - take[8] + orient_8_0 + orient_8_1 + orient_8_2
   + orient_8_3 + orient_8_4 + orient_8_5 = 0
 link_lx_8: lx[8] - 17 orient_8_0 - 18 orient_8_1 - 28 orient_8_2
   - 28 orient_8_3 - 18 orient_8_4 - 17 orient_8_5 = 0
 link_ly_8: ly[8] - 18 orient_8_0 - 17 orient_8_1 - 17 orient_8_2
   - 18 orient_8_3 - 28 orient_8_4 - 28 orient_8_5 = 0
 link_lz_8: lz[8] - 28 orient_8_0 - 28 orient_8_1 - 18 orient_8_2
   - 17 orient_8_3 - 17 orient_8_4 - 18 orient_8_5 = 0
 fit_x_8: - L + 200 take[8] + x[8] + lx[8] <= 200
 fit_y_8: - W + 200 take[8] + y[8] + ly[8] <= 200
 fit_z_8: - H + 200 take[8] + z[8] + lz[8] <= 200
 activate_x_8: - 200 take[8] + x[8] <= 0
 activate_y_8: - 200 take[8] + y[8] <= 0
 activate_z_8: - 200 take[8] + z[8] <= 0
 one_orientation_9: - take[9] + orient_9_0 + orient_9_1 + orient_9_2
   + orient_9_3 + orient_9_4 + orient_9_5 = 0
 link_lx_9: lx[9] - 32 orient_9_0 - 22 orient_9_1 - 19 orient_9_2
   - 32 orient_9_3 - 19 orient_9_4 - 22 orient_9_5 = 0
 link_ly_9: ly[9] - 19 orient_9_0 - 19 orient_9_1 - 22 orient_9_2
   - 22 orient_9_3 - 32 orient_9_4 - 32 orient_9_5 = 0
 link_lz_9: lz[9] - 22 orient_9_0 - 32 orient_9_1 - 32 orient_9_2
   - 19 orient_9_3 - 22 orient_9_4 - 19 orient_9_5 = 0
 fit_x_9: - L + 200 take[9] + x[9] + lx[9] <= 200
 fit_y_9: - W + 200 take[9] + y[9] + ly[9] <= 200
 fit_z_9: - H + 200 take[9] + z[9] + lz[9] <= 200
 activate_x_9: - 200 take[9] + x[9] <= 0
 activate_y_9: - 200 take[9] + y[9] <= 0
 activate_z_9: - 200 take[9] + z[9] <= 0
 nonoverlap_logic_0_1: - take[0] - take[1] + left_0_1 + left_1_0
   + front_0_1 + front_1_0 + below_0_1 + below_1_0 >= -1
 sep_x1_0_1: x[0] - x[1] + lx[0] + 200 left_0_1 <= 200
 sep_x2_0_1: - x[0] + x[1] + lx[1] + 200 left_1_0 <= 200
 sep_y1_0_1: y[0] - y[1] + ly[0] + 200 front_0_1 <= 200
 sep_y2_0_1: - y[0] + y[1] + ly[1] + 200 front_1_0 <= 200
 sep_z1_0_1: z[0] - z[1] + lz[0] + 200 below_0_1 <= 200
 sep_z2_0_1: - z[0] + z[1] + lz[1] + 200 below_1_0 <= 200
 nonoverlap_logic_0_2: - take[0] - take[2] + left_0_2 + left_2_0
   + front_0_2 + front_2_0 + below_0_2 + below_2_0 >= -1
 sep_x1_0_2: x[0] - x[2] + lx[0] + 200 left_0_2 <= 200
 sep_x2_0_2: - x[0] + x[2] + lx[2] + 200 left_2_0 <= 200
 sep_y1_0_2: y[0] - y[2] + ly[0] + 200 front_0_2 <= 200
 sep_y2_0_2: - y[0] + y[2] + ly[2] + 200 front_2_0 <= 200
 sep_z1_0_2: z[0] - z[2] + lz[0] + 200 below_0_2 <= 200
 sep_z2_0_2: - z[0] + z[2] + lz[2] + 200 below_2_0 <= 200
 nonoverlap_logic_0_3: - take[0] - take[3] + left_0_3 + left_3_0
   + front_0_3 + front_3_0 + below_0_3 + below_3_0 >= -1
 sep_x1_0_3: x[0] - x[3] + lx[0] + 200 left_0_3 <= 200
 sep_x2_0_3: - x[0] + x[3] + lx[3] + 200 left_3_0 <= 200
 sep_y1_0_3: y[0] - y[3] + ly[0] + 200 front_0_3 <= 200
 sep_y2_0_3: - y[0] + y[3] + ly[3] + 200 front_3_0 <= 200
 sep_z1_0_3: z[0] - z[3] + lz[0] + 200 below_0_3 <= 200
 sep_z2_0_3: - z[0] + z[3] + lz[3] + 200 below_3_0 <= 200
 nonoverlap_logic_0_4: - take[0] - take[4] + left_0_4 + left_4_0
   + front_0_4 + front_4_0 + below_0_4 + below_4_0 >= -1
 sep_x1_0_4: x[0] - x[4] + lx[0] + 200 left_0_4 <= 200
 sep_x2_0_4: - x[0] + x[4] + lx[4] + 200 left_4_0 <= 200
 sep_y1_0_4: y[0] - y[4] + ly[0] + 200 front_0_4 <= 200
 sep_y2_0_4: - y[0] + y[4] + ly[4] + 200 front_4_0 <= 200
 sep_z1_0_4: z[0] - z[4] + lz[0] + 200 below_0_4 <= 200
 sep_z2_0_4: - z[0] + z[4] + lz[4] + 200 below_4_0 <= 200
 nonoverlap_logic_0_5: - take[0] - take[5] + left_0_5 + left_5_0
   + front_0_5 + front_5_0 + below_0_5 + below_5_0 >= -1
 sep_x1_0_5: x[0] - x[5] + lx[0] + 200 left_0_5 <= 200
 sep_x2_0_5: - x[0] + x[5] + lx[5] + 200 left_5_0 <= 200
 sep_y1_0_5: y[0] - y[5] + ly[0] + 200 front_0_5 <= 200
 sep_y2_0_5: - y[0] + y[5] + ly[5] + 200 front_5_0 <= 200
 sep_z1_0_5: z[0] - z[5] + lz[0] + 200 below_0_5 <= 200
 sep_z2_0_5: - z[0] + z[5] + lz[5] + 200 below_5_0 <= 200
 nonoverlap_logic_0_6: - take[0] - take[6] + left_0_6 + left_6_0
   + front_0_6 + front_6_0 + below_0_6 + below_6_0 >= -1
 sep_x1_0_6: x[0] - x[6] + lx[0] + 200 left_0_6 <= 200
 sep_x2_0_6: - x[0] + x[6] + lx[6] + 200 left_6_0 <= 200
 sep_y1_0_6: y[0] - y[6] + ly[0] + 200 front_0_6 <= 200
 sep_y2_0_6: - y[0] + y[6] + ly[6] + 200 front_6_0 <= 200
 sep_z1_0_6: z[0] - z[6] + lz[0] + 200 below_0_6 <= 200
 sep_z2_0_6: - z[0] + z[6] + lz[6] + 200 below_6_0 <= 200
 nonoverlap_logic_0_7: - take[0] - take[7] + left_0_7 + left_7_0
   + front_0_7 + front_7_0 + below_0_7 + below_7_0 >= -1
 sep_x1_0_7: x[0] - x[7] + lx[0] + 200 left_0_7 <= 200
 sep_x2_0_7: - x[0] + x[7] + lx[7] + 200 left_7_0 <= 200
 sep_y1_0_7: y[0] - y[7] + ly[0] + 200 front_0_7 <= 200
 sep_y2_0_7: - y[0] + y[7] + ly[7] + 200 front_7_0 <= 200
 sep_z1_0_7: z[0] - z[7] + lz[0] + 200 below_0_7 <= 200
 sep_z2_0_7: - z[0] + z[7] + lz[7] + 200 below_7_0 <= 200
 nonoverlap_logic_0_8: - take[0] - take[8] + left_0_8 + left_8_0
   + front_0_8 + front_8_0 + below_0_8 + below_8_0 >= -1
 sep_x1_0_8: x[0] - x[8] + lx[0] + 200 left_0_8 <= 200
 sep_x2_0_8: - x[0] + x[8] + lx[8] + 200 left_8_0 <= 200
 sep_y1_0_8: y[0] - y[8] + ly[0] + 200 front_0_8 <= 200
 sep_y2_0_8: - y[0] + y[8] + ly[8] + 200 front_8_0 <= 200
 sep_z1_0_8: z[0] - z[8] + lz[0] + 200 below_0_8 <= 200
 sep_z2_0_8: - z[0] + z[8] + lz[8] + 200 below_8_0 <= 200
 nonoverlap_logic_0_9: - take[0] - take[9] + left_0_9 + left_9_0
   + front_0_9 + front_9_0 + below_0_9 + below_9_0 >= -1
 sep_x1_0_9: x[0] - x[9] + lx[0] + 200 left_0_9 <= 200
 sep_x2_0_9: - x[0] + x[9] + lx[9] + 200 left_9_0 <= 200
 sep_y1_0_9: y[0] - y[9] + ly[0] + 200 front_0_9 <= 200
 sep_y2_0_9: - y[0] + y[9] + ly[9] + 200 front_9_0 <= 200
 sep_z1_0_9: z[0] - z[9] + lz[0] + 200 below_0_9 <= 200
 sep_z2_0_9: - z[0] + z[9] + lz[9] + 200 below_9_0 <= 200
 nonoverlap_logic_1_2: - take[1] - take[2] + left_1_2 + left_2_1
   + front_1_2 + front_2_1 + below_1_2 + below_2_1 >= -1
 sep_x1_1_2: x[1] - x[2] + lx[1] + 200 left_1_2 <= 200
 sep_x2_1_2: - x[1] + x[2] + lx[2] + 200 left_2_1 <= 200
 sep_y1_1_2: y[1] - y[2] + ly[1] + 200 front_1_2 <= 200
 sep_y2_1_2: - y[1] + y[2] + ly[2] + 200 front_2_1 <= 200
 sep_z1_1_2: z[1] - z[2] + lz[1] + 200 below_1_2 <= 200
 sep_z2_1_2: - z[1] + z[2] + lz[2] + 200 below_2_1 <= 200
 nonoverlap_logic_1_3: - take[1] - take[3] + left_1_3 + left_3_1
   + front_1_3 + front_3_1 + below_1_3 + below_3_1 >= -1
 sep_x1_1_3: x[1] - x[3] + lx[1] + 200 left_1_3 <= 200
 sep_x2_1_3: - x[1] + x[3] + lx[3] + 200 left_3_1 <= 200
 sep_y1_1_3: y[1] - y[3] + ly[1] + 200 front_1_3 <= 200
 sep_y2_1_3: - y[1] + y[3] + ly[3] + 200 front_3_1 <= 200
 sep_z1_1_3: z[1] - z[3] + lz[1] + 200 below_1_3 <= 200
 sep_z2_1_3: - z[1] + z[3] + lz[3] + 200 below_3_1 <= 200
 nonoverlap_logic_1_4: - take[1] - take[4] + left_1_4 + left_4_1
   + front_1_4 + front_4_1 + below_1_4 + below_4_1 >= -1
 sep_x1_1_4: x[1] - x[4] + lx[1] + 200 left_1_4 <= 200
 sep_x2_1_4: - x[1] + x[4] + lx[4] + 200 left_4_1 <= 200
 sep_y1_1_4: y[1] - y[4] + ly[1] + 200 front_1_4 <= 200
 sep_y2_1_4: - y[1] + y[4] + ly[4] + 200 front_4_1 <= 200
 sep_z1_1_4: z[1] - z[4] + lz[1] + 200 below_1_4 <= 200
 sep_z2_1_4: - z[1] + z[4] + lz[4] + 200 below_4_1 <= 200
 nonoverlap_logic_1_5: - take[1] - take[5] + left_1_5 + left_5_1
   + front_1_5 + front_5_1 + below_1_5 + below_5_1 >= -1
 sep_x1_1_5: x[1] - x[5] + lx[1] + 200 left_1_5 <= 200
 sep_x2_1_5: - x[1] + x[5] + lx[5] + 200 left_5_1 <= 200
 sep_y1_1_5: y[1] - y[5] + ly[1] + 200 front_1_5 <= 200
 sep_y2_1_5: - y[1] + y[5] + ly[5] + 200 front_5_1 <= 200
 sep_z1_1_5: z[1] - z[5] + lz[1] + 200 below_1_5 <= 200
 sep_z2_1_5: - z[1] + z[5] + lz[5] + 200 below_5_1 <= 200
 nonoverlap_logic_1_6: - take[1] - take[6] + left_1_6 + left_6_1
   + front_1_6 + front_6_1 + below_1_6 + below_6_1 >= -1
 sep_x1_1_6: x[1] - x[6] + lx[1] + 200 left_1_6 <= 200
 sep_x2_1_6: - x[1] + x[6] + lx[6] + 200 left_6_1 <= 200
 sep_y1_1_6: y[1] - y[6] + ly[1] + 200 front_1_6 <= 200
 sep_y2_1_6: - y[1] + y[6] + ly[6] + 200 front_6_1 <= 200
 sep_z1_1_6: z[1] - z[6] + lz[1] + 200 below_1_6 <= 200
 sep_z2_1_6: - z[1] + z[6] + lz[6] + 200 below_6_1 <= 200
 nonoverlap_logic_1_7: - take[1] - take[7] + left_1_7 + left_7_1
   + front_1_7 + front_7_1 + below_1_7 + below_7_1 >= -1
 sep_x1_1_7: x[1] - x[7] + lx[1] + 200 left_1_7 <= 200
 sep_x2_1_7: - x[1] + x[7] + lx[7] + 200 left_7_1 <= 200
 sep_y1_1_7: y[1] - y[7] + ly[1] + 200 front_1_7 <= 200
 sep_y2_1_7: - y[1] + y[7] + ly[7] + 200 front_7_1 <= 200
 sep_z1_1_7: z[1] - z[7] + lz[1] + 200 below_1_7 <= 200
 sep_z2_1_7: - z[1] + z[7] + lz[7] + 200 below_7_1 <= 200
 nonoverlap_logic_1_8: - take[1] - take[8] + left_1_8 + left_8_1
   + front_1_8 + front_8_1 + below_1_8 + below_8_1 >= -1
 sep_x1_1_8: x[1] - x[8] + lx[1] + 200 left_1_8 <= 200
 sep_x2_1_8: - x[1] + x[8] + lx[8] + 200 left_8_1 <= 200
 sep_y1_1_8: y[1] - y[8] + ly[1] + 200 front_1_8 <= 200
 sep_y2_1_8: - y[1] + y[8] + ly[8] + 200 front_8_1 <= 200
 sep_z1_1_8: z[1] - z[8] + lz[1] + 200 below_1_8 <= 200
 sep_z2_1_8: - z[1] + z[8] + lz[8] + 200 below_8_1 <= 200
 nonoverlap_logic_1_9: - take[1] - take[9] + left_1_9 + left_9_1
   + front_1_9 + front_9_1 + below_1_9 + below_9_1 >= -1
 sep_x1_1_9: x[1] - x[9] + lx[1] + 200 left_1_9 <= 200
 sep_x2_1_9: - x[1] + x[9] + lx[9] + 200 left_9_1 <= 200
 sep_y1_1_9: y[1] - y[9] + ly[1] + 200 front_1_9 <= 200
 sep_y2_1_9: - y[1] + y[9] + ly[9] + 200 front_9_1 <= 200
 sep_z1_1_9: z[1] - z[9] + lz[1] + 200 below_1_9 <= 200
 sep_z2_1_9: - z[1] + z[9] + lz[9] + 200 below_9_1 <= 200
 nonoverlap_logic_2_3: - take[2] - take[3] + left_2_3 + left_3_2
   + front_2_3 + front_3_2 + below_2_3 + below_3_2 >= -1
 sep_x1_2_3: x[2] - x[3] + lx[2] + 200 left_2_3 <= 200
 sep_x2_2_3: - x[2] + x[3] + lx[3] + 200 left_3_2 <= 200
 sep_y1_2_3: y[2] - y[3] + ly[2] + 200 front_2_3 <= 200
 sep_y2_2_3: - y[2] + y[3] + ly[3] + 200 front_3_2 <= 200
 sep_z1_2_3: z[2] - z[3] + lz[2] + 200 below_2_3 <= 200
 sep_z2_2_3: - z[2] + z[3] + lz[3] + 200 below_3_2 <= 200
 nonoverlap_logic_2_4: - take[2] - take[4] + left_2_4 + left_4_2
   + front_2_4 + front_4_2 + below_2_4 + below_4_2 >= -1
 sep_x1_2_4: x[2] - x[4] + lx[2] + 200 left_2_4 <= 200
 sep_x2_2_4: - x[2] + x[4] + lx[4] + 200 left_4_2 <= 200
 sep_y1_2_4: y[2] - y[4] + ly[2] + 200 front_2_4 <= 200
 sep_y2_2_4: - y[2] + y[4] + ly[4] + 200 front_4_2 <= 200
 sep_z1_2_4: z[2] - z[4] + lz[2] + 200 below_2_4 <= 200
 sep_z2_2_4: - z[2] + z[4] + lz[4] + 200 below_4_2 <= 200
 nonoverlap_logic_2_5: - take[2] - take[5] + left_2_5 + left_5_2
   + front_2_5 + front_5_2 + below_2_5 + below_5_2 >= -1
 sep_x1_2_5: x[2] - x[5] + lx[2] + 200 left_2_5 <= 200
 sep_x2_2_5: - x[2] + x[5] + lx[5] + 200 left_5_2 <= 200
 sep_y1_2_5: y[2] - y[5] + ly[2] + 200 front_2_5 <= 200
 sep_y2_2_5: - y[2] + y[5] + ly[5] + 200 front_5_2 <= 200
 sep_z1_2_5: z[2] - z[5] + lz[2] + 200 below_2_5 <= 200
 sep_z2_2_5: - z[2] + z[5] + lz[5] + 200 below_5_2 <= 200
 nonoverlap_logic_2_6: - take[2] - take[6] + left_2_6 + left_6_2
   + front_2_6 + front_6_2 + below_2_6 + below_6_2 >= -1
 sep_x1_2_6: x[2] - x[6] + lx[2] + 200 left_2_6 <= 200
 sep_x2_2_6: - x[2] + x[6] + lx[6] + 200 left_6_2 <= 200
 sep_y1_2_6: y[2] - y[6] + ly[2] + 200 front_2_6 <= 200
 sep_y2_2_6: - y[2] + y[6] + ly[6] + 200 front_6_2 <= 200
 sep_z1_2_6: z[2] - z[6] + lz[2] + 200 below_2_6 <= 200
 sep_z2_2_6: - z[2] + z[6] + lz[6] + 200 below_6_2 <= 200
 nonoverlap_logic_2_7: - take[2] - take[7] + left_2_7 + left_7_2
   + front_2_7 + front_7_2 + below_2_7 + below_7_2 >= -1
 sep_x1_2_7: x[2] - x[7] + lx[2] + 200 left_2_7 <= 200
 sep_x2_2_7: - x[2] + x[7] + lx[7] + 200 left_7_2 <= 200
 sep_y1_2_7: y[2] - y[7] + ly[2] + 200 front_2_7 <= 200
 sep_y2_2_7: - y[2] + y[7] + ly[7] + 200 front_7_2 <= 200
 sep_z1_2_7: z[2] - z[7] + lz[2] + 200 below_2_7 <= 200
 sep_z2_2_7: - z[2] + z[7] + lz[7] + 200 below_7_2 <= 200
 nonoverlap_logic_2_8: - take[2] - take[8] + left_2_8 + left_8_2
   + front_2_8 + front_8_2 + below_2_8 + below_8_2 >= -1
 sep_x1_2_8: x[2] - x[8] + lx[2] + 200 left_2_8 <= 200
 sep_x2_2_8: - x[2] + x[8] + lx[8] + 200 left_8_2 <= 200
 sep_y1_2_8: y[2] - y[8] + ly[2] + 200 front_2_8 <= 200
 sep_y2_2_8: - y[2] + y[8] + ly[8] + 200 front_8_2 <= 200
 sep_z1_2_8: z[2] - z[8] + lz[2] + 200 below_2_8 <= 200
 sep_z2_2_8: - z[2] + z[8] + lz[8] + 200 below_8_2 <= 200
 nonoverlap_logic_2_9: - take[2] - take[9] + left_2_9 + left_9_2
   + front_2_9 + front_9_2 + below_2_9 + below_9_2 >= -1
 sep_x1_2_9: x[2] - x[9] + lx[2] + 200 left_2_9 <= 200
 sep_x2_2_9: - x[2] + x[9] + lx[9] + 200 left_9_2 <= 200
 sep_y1_2_9: y[2] - y[9] + ly[2] + 200 front_2_9 <= 200
 sep_y2_2_9: - y[2] + y[9] + ly[9] + 200 front_9_2 <= 200
 sep_z1_2_9: z[2] - z[9] + lz[2] + 200 below_2_9 <= 200
 sep_z2_2_9: - z[2] + z[9] + lz[9] + 200 below_9_2 <= 200
 nonoverlap_logic_3_4: - take[3] - take[4] + left_3_4 + left_4_3
   + front_3_4 + front_4_3 + below_3_4 + below_4_3 >= -1
 sep_x1_3_4: x[3] - x[4] + lx[3] + 200 left_3_4 <= 200
 sep_x2_3_4: - x[3] + x[4] + lx[4] + 200 left_4_3 <= 200
 sep_y1_3_4: y[3] - y[4] + ly[3] + 200 front_3_4 <= 200
 sep_y2_3_4: - y[3] + y[4] + ly[4] + 200 front_4_3 <= 200
 sep_z1_3_4: z[3] - z[4] + lz[3] + 200 below_3_4 <= 200
 sep_z2_3_4: - z[3] + z[4] + lz[4] + 200 below_4_3 <= 200
 nonoverlap_logic_3_5: - take[3] - take[5] + left_3_5 + left_5_3
   + front_3_5 + front_5_3 + below_3_5 + below_5_3 >= -1
 sep_x1_3_5: x[3] - x[5] + lx[3] + 200 left_3_5 <= 200
 sep_x2_3_5: - x[3] + x[5] + lx[5] + 200 left_5_3 <= 200
 sep_y1_3_5: y[3] - y[5] + ly[3] + 200 front_3_5 <= 200
 sep_y2_3_5: - y[3] + y[5] + ly[5] + 200 front_5_3 <= 200
 sep_z1_3_5: z[3] - z[5] + lz[3] + 200 below_3_5 <= 200
 sep_z2_3_5: - z[3] + z[5] + lz[5] + 200 below_5_3 <= 200
 nonoverlap_logic_3_6: - take[3] - take[6] + left_3_6 + left_6_3
   + front_3_6 + front_6_3 + below_3_6 + below_6_3 >= -1
 sep_x1_3_6: x[3] - x[6] + lx[3] + 200 left_3_6 <= 200
 sep_x2_3_6: - x[3] + x[6] + lx[6] + 200 left_6_3 <= 200
 sep_y1_3_6: y[3] - y[6] + ly[3] + 200 front_3_6 <= 200
 sep_y2_3_6: - y[3] + y[6] + ly[6] + 200 front_6_3 <= 200
 sep_z1_3_6: z[3] - z[6] + lz[3] + 200 below_3_6 <= 200
 sep_z2_3_6: - z[3] + z[6] + lz[6] + 200 below_6_3 <= 200
 nonoverlap_logic_3_7: - take[3] - take[7] + left_3_7 + left_7_3
   + front_3_7 + front_7_3 + below_3_7 + below_7_3 >= -1
 sep_x1_3_7: x[3] - x[7] + lx[3] + 200 left_3_7 <= 200
 sep_x2_3_7: - x[3] + x[7] + lx[7] + 200 left_7_3 <= 200
 sep_y1_3_7: y[3] - y[7] + ly[3] + 200 front_3_7 <= 200
 sep_y2_3_7: - y[3] + y[7] + ly[7] + 200 front_7_3 <= 200
 sep_z1_3_7: z[3] - z[7] + lz[3] + 200 below_3_7 <= 200
 sep_z2_3_7: - z[3] + z[7] + lz[7] + 200 below_7_3 <= 200
 nonoverlap_logic_3_8: - take[3] - take[8] + left_3_8 + left_8_3
   + front_3_8 + front_8_3 + below_3_8 + below_8_3 >= -1
 sep_x1_3_8: x[3] - x[8] + lx[3] + 200 left_3_8 <= 200
 sep_x2_3_8: - x[3] + x[8] + lx[8] + 200 left_8_3 <= 200
 sep_y1_3_8: y[3] - y[8] + ly[3] + 200 front_3_8 <= 200
 sep_y2_3_8: - y[3] + y[8] + ly[8] + 200 front_8_3 <= 200
 sep_z1_3_8: z[3] - z[8] + lz[3] + 200 below_3_8 <= 200
 sep_z2_3_8: - z[3] + z[8] + lz[8] + 200 below_8_3 <= 200
 nonoverlap_logic_3_9: - take[3] - take[9] + left_3_9 + left_9_3
   + front_3_9 + front_9_3 + below_3_9 + below_9_3 >= -1
 sep_x1_3_9: x[3] - x[9] + lx[3] + 200 left_3_9 <= 200
 sep_x2_3_9: - x[3] + x[9] + lx[9] + 200 left_9_3 <= 200
 sep_y1_3_9: y[3] - y[9] + ly[3] + 200 front_3_9 <= 200
 sep_y2_3_9: - y[3] + y[9] + ly[9] + 200 front_9_3 <= 200
 sep_z1_3_9: z[3] - z[9] + lz[3] + 200 below_3_9 <= 200
 sep_z2_3_9: - z[3] + z[9] + lz[9] + 200 below_9_3 <= 200
 nonoverlap_logic_4_5: - take[4] - take[5] + left_4_5 + left_5_4
   + front_4_5 + front_5_4 + below_4_5 + below_5_4 >= -1
 sep_x1_4_5: x[4] - x[5] + lx[4] + 200 left_4_5 <= 200
 sep_x2_4_5: - x[4] + x[5] + lx[5] + 200 left_5_4 <= 200
 sep_y1_4_5: y[4] - y[5] + ly[4] + 200 front_4_5 <= 200
 sep_y2_4_5: - y[4] + y[5] + ly[5] + 200 front_5_4 <= 200
 sep_z1_4_5: z[4] - z[5] + lz[4] + 200 below_4_5 <= 200
 sep_z2_4_5: - z[4] + z[5] + lz[5] + 200 below_5_4 <= 200
 nonoverlap_logic_4_6: - take[4] - take[6] + left_4_6 + left_6_4
   + front_4_6 + front_6_4 + below_4_6 + below_6_4 >= -1
 sep_x1_4_6: x[4] - x[6] + lx[4] + 200 left_4_6 <= 200
 sep_x2_4_6: - x[4] + x[6] + lx[6] + 200 left_6_4 <= 200
 sep_y1_4_6: y[4] - y[6] + ly[4] + 200 front_4_6 <= 200
 sep_y2_4_6: - y[4] + y[6] + ly[6] + 200 front_6_4 <= 200
 sep_z1_4_6: z[4] - z[6] + lz[4] + 200 below_4_6 <= 200
 sep_z2_4_6: - z[4] + z[6] + lz[6] + 200 below_6_4 <= 200
 nonoverlap_logic_4_7: - take[4] - take[7] + left_4_7 + left_7_4
   + front_4_7 + front_7_4 + below_4_7 + below_7_4 >= -1
 sep_x1_4_7: x[4] - x[7] + lx[4] + 200 left_4_7 <= 200
 sep_x2_4_7: - x[4] + x[7] + lx[7] + 200 left_7_4 <= 200
 sep_y1_4_7: y[4] - y[7] + ly[4] + 200 front_4_7 <= 200
 sep_y2_4_7: - y[4] + y[7] + ly[7] + 200 front_7_4 <= 200
 sep_z1_4_7: z[4] - z[7] + lz[4] + 200 below_4_7 <= 200
 sep_z2_4_7: - z[4] + z[7] + lz[7] + 200 below_7_4 <= 200
 nonoverlap_logic_4_8: - take[4] - take[8] + left_4_8 + left_8_4
   + front_4_8 + front_8_4 + below_4_8 + below_8_4 >= -1
 sep_x1_4_8: x[4] - x[8] + lx[4] + 200 left_4_8 <= 200
 sep_x2_4_8: - x[4] + x[8] + lx[8] + 200 left_8_4 <= 200
 sep_y1_4_8: y[4] - y[8] + ly[4] + 200 front_4_8 <= 200
 sep_y2_4_8: - y[4] + y[8] + ly[8] + 200 front_8_4 <= 200
 sep_z1_4_8: z[4] - z[8] + lz[4] + 200 below_4_8 <= 200
 sep_z2_4_8: - z[4] + z[8] + lz[8] + 200 below_8_4 <= 200
 nonoverlap_logic_4_9: - take[4] - take[9] + left_4_9 + left_9_4
   + front_4_9 + front_9_4 + below_4_9 + below_9_4 >= -1
 sep_x1_4_9: x[4] - x[9] + lx[4] + 200 left_4_9 <= 200
 sep_x2_4_9: - x[4] + x[9] + lx[9] + 200 left_9_4 <= 200
 sep_y1_4_9: y[4] - y[9] + ly[4] + 200 front_4_9 <= 200
 sep_y2_4_9: - y[4] + y[9] + ly[9] + 200 front_9_4 <= 200
 sep_z1_4_9: z[4] - z[9] + lz[4] + 200 below_4_9 <= 200
 sep_z2_4_9: - z[4] + z[9] + lz[9] + 200 below_9_4 <= 200
 nonoverlap_logic_5_6: - take[5] - take[6] + left_5_6 + left_6_5
   + front_5_6 + front_6_5 + below_5_6 + below_6_5 >= -1
 sep_x1_5_6: x[5] - x[6] + lx[5] + 200 left_5_6 <= 200
 sep_x2_5_6: - x[5] + x[6] + lx[6] + 200 left_6_5 <= 200
 sep_y1_5_6: y[5] - y[6] + ly[5] + 200 front_5_6 <= 200
 sep_y2_5_6: - y[5] + y[6] + ly[6] + 200 front_6_5 <= 200
 sep_z1_5_6: z[5] - z[6] + lz[5] + 200 below_5_6 <= 200
 sep_z2_5_6: - z[5] + z[6] + lz[6] + 200 below_6_5 <= 200
 nonoverlap_logic_5_7: - take[5] - take[7] + left_5_7 + left_7_5
   + front_5_7 + front_7_5 + below_5_7 + below_7_5 >= -1
 sep_x1_5_7: x[5] - x[7] + lx[5] + 200 left_5_7 <= 200
 sep_x2_5_7: - x[5] + x[7] + lx[7] + 200 left_7_5 <= 200
 sep_y1_5_7: y[5] - y[7] + ly[5] + 200 front_5_7 <= 200
 sep_y2_5_7: - y[5] + y[7] + ly[7] + 200 front_7_5 <= 200
 sep_z1_5_7: z[5] - z[7] + lz[5] + 200 below_5_7 <= 200
 sep_z2_5_7: - z[5] + z[7] + lz[7] + 200 below_7_5 <= 200
 nonoverlap_logic_5_8: - take[5] - take[8] + left_5_8 + left_8_5
   + front_5_8 + front_8_5 + below_5_8 + below_8_5 >= -1
 sep_x1_5_8: x[5] - x[8] + lx[5] + 200 left_5_8 <= 200
 sep_x2_5_8: - x[5] + x[8] + lx[8] + 200 left_8_5 <= 200
 sep_y1_5_8: y[5] - y[8] + ly[5] + 200 front_5_8 <= 200
 sep_y2_5_8: - y[5] + y[8] + ly[8] + 200 front_8_5 <= 200
 sep_z1_5_8: z[5] - z[8] + lz[5] + 200 below_5_8 <= 200
 sep_z2_5_8: - z[5] + z[8] + lz[8] + 200 below_8_5 <= 200
 nonoverlap_logic_5_9: - take[5] - take[9] + left_5_9 + left_9_5
   + front_5_9 + front_9_5 + below_5_9 + below_9_5 >= -1
 sep_x1_5_9: x[5] - x[9] + lx[5] + 200 left_5_9 <= 200
 sep_x2_5_9: - x[5] + x[9] + lx[9] + 200 left_9_5 <= 200
 sep_y1_5_9: y[5] - y[9] + ly[5] + 200 front_5_9 <= 200
 sep_y2_5_9: - y[5] + y[9] + ly[9] + 200 front_9_5 <= 200
 sep_z1_5_9: z[5] - z[9] + lz[5] + 200 below_5_9 <= 200
 sep_z2_5_9: - z[5] + z[9] + lz[9] + 200 below_9_5 <= 200
 nonoverlap_logic_6_7: - take[6] - take[7] + left_6_7 + left_7_6
   + front_6_7 + front_7_6 + below_6_7 + below_7_6 >= -1
 sep_x1_6_7: x[6] - x[7] + lx[6] + 200 left_6_7 <= 200
 sep_x2_6_7: - x[6] + x[7] + lx[7] + 200 left_7_6 <= 200
 sep_y1_6_7: y[6] - y[7] + ly[6] + 200 front_6_7 <= 200
 sep_y2_6_7: - y[6] + y[7] + ly[7] + 200 front_7_6 <= 200
 sep_z1_6_7: z[6] - z[7] + lz[6] + 200 below_6_7 <= 200
 sep_z2_6_7: - z[6] + z[7] + lz[7] + 200 below_7_6 <= 200
 nonoverlap_logic_6_8: - take[6] - take[8] + left_6_8 + left_8_6
   + front_6_8 + front_8_6 + below_6_8 + below_8_6 >= -1
 sep_x1_6_8: x[6] - x[8] + lx[6] + 200 left_6_8 <= 200
 sep_x2_6_8: - x[6] + x[8] + lx[8] + 200 left_8_6 <= 200
 sep_y1_6_8: y[6] - y[8] + ly[6] + 200 front_6_8 <= 200
 sep_y2_6_8: - y[6] + y[8] + ly[8] + 200 front_8_6 <= 200
 sep_z1_6_8: z[6] - z[8] + lz[6] + 200 below_6_8 <= 200
 sep_z2_6_8: - z[6] + z[8] + lz[8] + 200 below_8_6 <= 200
 nonoverlap_logic_6_9: - take[6] - take[9] + left_6_9 + left_9_6
   + front_6_9 + front_9_6 + below_6_9 + below_9_6 >= -1
 sep_x1_6_9: x[6] - x[9] + lx[6] + 200 left_6_9 <= 200
 sep_x2_6_9: - x[6] + x[9] + lx[9] + 200 left_9_6 <= 200
 sep_y1_6_9: y[6] - y[9] + ly[6] + 200 front_6_9 <= 200
 sep_y2_6_9: - y[6] + y[9] + ly[9] + 200 front_9_6 <= 200
 sep_z1_6_9: z[6] - z[9] + lz[6] + 200 below_6_9 <= 200
 sep_z2_6_9: - z[6] + z[9] + lz[9] + 200 below_9_6 <= 200
 nonoverlap_logic_7_8: - take[7] - take[8] + left_7_8 + left_8_7
   + front_7_8 + front_8_7 + below_7_8 + below_8_7 >= -1
 sep_x1_7_8: x[7] - x[8] + lx[7] + 200 left_7_8 <= 200
 sep_x2_7_8: - x[7] + x[8] + lx[8] + 200 left_8_7 <= 200
 sep_y1_7_8: y[7] - y[8] + ly[7] + 200 front_7_8 <= 200
 sep_y2_7_8: - y[7] + y[8] + ly[8] + 200 front_8_7 <= 200
 sep_z1_7_8: z[7] - z[8] + lz[7] + 200 below_7_8 <= 200
 sep_z2_7_8: - z[7] + z[8] + lz[8] + 200 below_8_7 <= 200
 nonoverlap_logic_7_9: - take[7] - take[9] + left_7_9 + left_9_7
   + front_7_9 + front_9_7 + below_7_9 + below_9_7 >= -1
 sep_x1_7_9: x[7] - x[9] + lx[7] + 200 left_7_9 <= 200
 sep_x2_7_9: - x[7] + x[9] + lx[9] + 200 left_9_7 <= 200
 sep_y1_7_9: y[7] - y[9] + ly[7] + 200 front_7_9 <= 200
 sep_y2_7_9: - y[7] + y[9] + ly[9] + 200 front_9_7 <= 200
 sep_z1_7_9: z[7] - z[9] + lz[7] + 200 below_7_9 <= 200
 sep_z2_7_9: - z[7] + z[9] + lz[9] + 200 below_9_7 <= 200
 nonoverlap_logic_8_9: - take[8] - take[9] + left_8_9 + left_9_8
   + front_8_9 + front_9_8 + below_8_9 + below_9_8 >= -1
 sep_x1_8_9: x[8] - x[9] + lx[8] + 200 left_8_9 <= 200
 sep_x2_8_9: - x[8] + x[9] + lx[9] + 200 left_9_8 <= 200
 sep_y1_8_9: y[8] - y[9] + ly[8] + 200 front_8_9 <= 200
 sep_y2_8_9: - y[8] + y[9] + ly[9] + 200 front_9_8 <= 200
 sep_z1_8_9: z[8] - z[9] + lz[8] + 200 below_8_9 <= 200
 sep_z2_8_9: - z[8] + z[9] + lz[9] + 200 below_9_8 <= 200
Bounds
 1 <= L <= 158
 1 <= W <= 158
 1 <= H <= 158
Binaries
 take[0] take[1] take[2] take[3] take[4] take[5] take[6] take[7] take[8]
 take[9] orient_0_0 orient_0_1 orient_0_2 orient_0_3 orient_0_4 orient_0_5
 orient_1_0 orient_1_1 orient_1_2 orient_1_3 orient_1_4 orient_1_5
 orient_2_0 orient_2_1 orient_2_2 orient_2_3 orient_2_4 orient_2_5
 orient_3_0 orient_3_1 orient_3_2 orient_3_3 orient_3_4 orient_3_5
 orient_4_0 orient_4_1 orient_4_2 orient_4_3 orient_4_4 orient_4_5
 orient_5_0 orient_5_1 orient_5_2 orient_5_3 orient_5_4 orient_5_5
 orient_6_0 orient_6_1 orient_6_2 orient_6_3 orient_6_4 orient_6_5
 orient_7_0 orient_7_1 orient_7_2 orient_7_3 orient_7_4 orient_7_5
 orient_8_0 orient_8_1 orient_8_2 orient_8_3 orient_8_4 orient_8_5
 orient_9_0 orient_9_1 orient_9_2 orient_9_3 orient_9_4 orient_9_5 left_0_1
 left_1_0 front_0_1 front_1_0 below_0_1 below_1_0 left_0_2 left_2_0
 front_0_2 front_2_0 below_0_2 below_2_0 left_0_3 left_3_0 front_0_3
 front_3_0 below_0_3 below_3_0 left_0_4 left_4_0 front_0_4 front_4_0
 below_0_4 below_4_0 left_0_5 left_5_0 front_0_5 front_5_0 below_0_5
 below_5_0 left_0_6 left_6_0 front_0_6 front_6_0 below_0_6 below_6_0
 left_0_7 left_7_0 front_0_7 front_7_0 below_0_7 below_7_0 left_0_8
 left_8_0 front_0_8 front_8_0 below_0_8 below_8_0 left_0_9 left_9_0
 front_0_9 front_9_0 below_0_9 below_9_0 left_1_2 left_2_1 front_1_2
 front_2_1 below_1_2 below_2_1 left_1_3 left_3_1 front_1_3 front_3_1
 below_1_3 below_3_1 left_1_4 left_4_1 front_1_4 front_4_1 below_1_4
 below_4_1 left_1_5 left_5_1 front_1_5 front_5_1 below_1_5 below_5_1
 left_1_6 left_6_1 front_1_6 front_6_1 below_1_6 below_6_1 left_1_7
 left_7_1 front_1_7 front_7_1 below_1_7 below_7_1 left_1_8 left_8_1
 front_1_8 front_8_1 below_1_8 below_8_1 left_1_9 left_9_1 front_1_9
 front_9_1 below_1_9 below_9_1 left_2_3 left_3_2 front_2_3 front_3_2
 below_2_3 below_3_2 left_2_4 left_4_2 front_2_4 front_4_2 below_2_4
 below_4_2 left_2_5 left_5_2 front_2_5 front_5_2 below_2_5 below_5_2
 left_2_6 left_6_2 front_2_6 front_6_2 below_2_6 below_6_2 left_2_7
 left_7_2 front_2_7 front_7_2 below_2_7 below_7_2 left_2_8 left_8_2
 front_2_8 front_8_2 below_2_8 below_8_2 left_2_9 left_9_2 front_2_9
 front_9_2 below_2_9 below_9_2 left_3_4 left_4_3 front_3_4 front_4_3
 below_3_4 below_4_3 left_3_5 left_5_3 front_3_5 front_5_3 below_3_5
 below_5_3 left_3_6 left_6_3 front_3_6 front_6_3 below_3_6 below_6_3
 left_3_7 left_7_3 front_3_7 front_7_3 below_3_7 below_7_3 left_3_8
 left_8_3 front_3_8 front_8_3 below_3_8 below_8_3 left_3_9 left_9_3
 front_3_9 front_9_3 below_3_9 below_9_3 left_4_5 left_5_4 front_4_5
 front_5_4 below_4_5 below_5_4 left_4_6 left_6_4 front_4_6 front_6_4
 below_4_6 below_6_4 left_4_7 left_7_4 front_4_7 front_7_4 below_4_7
 below_7_4 left_4_8 left_8_4 front_4_8 front_8_4 below_4_8 below_8_4
 left_4_9 left_9_4 front_4_9 front_9_4 below_4_9 below_9_4 left_5_6
 left_6_5 front_5_6 front_6_5 below_5_6 below_6_5 left_5_7 left_7_5
 front_5_7 front_7_5 below_5_7 below_7_5 left_5_8 left_8_5 front_5_8
 front_8_5 below_5_8 below_8_5 left_5_9 left_9_5 front_5_9 front_9_5
 below_5_9 below_9_5 left_6_7 left_7_6 front_6_7 front_7_6 below_6_7
 below_7_6 left_6_8 left_8_6 front_6_8 front_8_6 below_6_8 below_8_6
 left_6_9 left_9_6 front_6_9 front_9_6 below_6_9 below_9_6 left_7_8
 left_8_7 front_7_8 front_8_7 below_7_8 below_8_7 left_7_9 left_9_7
 front_7_9 front_9_7 below_7_9 below_9_7 left_8_9 left_9_8 front_8_9
 front_9_8 below_8_9 below_9_8
Generals
 L W H
End
