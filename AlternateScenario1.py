################################################################
################################################################
################SANAYİ SEKTÖRÜ VERİMLİLİK ARTIŞI################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from CGE import CGE
import networkx as nx

SAM      = pd.read_excel("SHM.xlsx", index_col = "index")
SAMCFrun = pd.read_excel("SHMCFrun.xlsx", index_col = "index")

class CGEResults():

    def __init__(self, sec_code):

        ########### BAZ YIL ###################
        self.m1 = CGE(SAM)
        self.result1 = self.m1.SolveModel()
        #######################################
        

        ########### KARŞI OLGUSAL DENGE [VERİMLİLİK ARTIŞI] #######
        self.m2 = CGE(SAM)
        self.m2.DCOBar = 30.67
        self.m2.DNGBar = 8.5
        self.m2.epsilon /= 1.2
        self.m2.E_Energy = 40
        self.m2.Ar *= 1.2
        self.m2.Ab *= 1.2
        self.m2.A3*=1.16
        # self.m2.mu1 = 0
        # self.m2.mu2 = 0
        # self.m2.mu3 = 1
        ############################################################

        self.result2 = self.m2.SolveModel()

        self.sectors = {0: "Tarım", 1: "Hizmet", 2:"Sanayi", 3:"Rafineriler", 4:"BOTAŞ"}
        self.sec_code = sec_code

        self.EndoVarandParameters()
        self.MacroVariables()

        result_df = pd.DataFrame(columns = ["base", "cfrun", "%Change"], index   = self.m1.init_values_str)
        result_df.base = self.result1.x
        result_df.cfrun = self.result2.x
        result_df["%Change"] = (result_df.cfrun - result_df.base) / result_df.base * 100
        result_df.to_excel("Results.xlsx")

        parameter_df = pd.DataFrame(index = self.m1.parameters_str, columns = ["Values"])
        parameter_df.Values = self.m1.parameters

        endovar_df = pd.DataFrame(index = self.m1.init_values_str, columns = ["Values"])
        endovar_df.Values = self.m1.init_values

        self.U_base, self.U_cfrun = -1*self.result1.fun, -1*self.result2.fun

    def AxisLimits(self):

        #-------------------------------Axes 1------------------------------------
        self.axis1xmin = [self.D_base[self.sec_code] if self.D_base[self.sec_code] < self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]
        self.axis1xmax = [self.D_base[self.sec_code] if self.D_base[self.sec_code] > self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]

        self.axis1ymin = [self.M_base[self.sec_code] if self.M_base[self.sec_code] < self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]
        self.axis1ymax = [self.M_base[self.sec_code] if self.M_base[self.sec_code] > self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]

        if self.sec_code == 0 or self.sec_code == 1:

            self.ax1.set_xlim(xmin = self.axis1xmin - self.axis1xmin/5, xmax = self.axis1xmax + self.axis1xmax/7)
            self.ax1.set_ylim(ymin = 0, ymax = self.axis1ymax + self.axis1ymax*4 )

        else:

            self.ax1.set_xlim(xmin = self.axis1xmin - self.axis1xmin/2, xmax = self.axis1xmax + self.axis1xmax/2)
            self.ax1.set_ylim(ymin = self.axis1ymin - self.axis1ymin/2, ymax = self.axis1ymax + self.axis1ymax/2 )


        #-------------------------------Axes 2------------------------------------
        self.axis2xmin = [self.D_base[self.sec_code] if self.D_base[self.sec_code] < self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]
        self.axis2xmax = [self.D_base[self.sec_code] if self.D_base[self.sec_code] > self.D_cfrun[self.sec_code] else self.D_cfrun[self.sec_code]][0]

        self.axis2ymin = [self.E_base[self.sec_code] if self.E_base[self.sec_code] < self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]
        self.axis2ymax = [self.E_base[self.sec_code] if self.E_base[self.sec_code] > self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]

        if self.sec_code == 0 or self.sec_code == 1:

            self.ax2.set_xlim(xmin = self.axis2xmin - self.axis2xmin/5, xmax = self.axis2xmax + self.axis2xmax/7)
            self.ax2.set_ylim(ymin = 0, ymax = self.axis2ymax + self.axis2ymax*4 )

        else:

            self.ax2.set_xlim(xmin = self.axis2xmin - self.axis2xmin/2, xmax = self.axis2xmax + self.axis2xmax/2)
            self.ax2.set_ylim(ymin = self.axis2ymin - self.axis2ymin/2, ymax = self.axis2ymax + self.axis2ymax/2 )


        #-------------------------------Axes 3------------------------------------
        self.ForeignTradeXLineMaxValue = [self.E_base[self.sec_code] if self.E_base[self.sec_code] > self.E_cfrun[self.sec_code] else self.E_cfrun[self.sec_code]][0]
        self.ForeignTradeYLineMaxValue = [self.M_base[self.sec_code] if self.M_base[self.sec_code] > self.M_cfrun[self.sec_code] else self.M_cfrun[self.sec_code]][0]

        ax3ymin = [self.Sf_base[self.sec_code] if self.Sf_base[self.sec_code] <self.Sf_cfrun[self.sec_code] else self.Sf_cfrun[self.sec_code]][0]

        self.ax3.set_xlim(xmin = 0, xmax = self.ForeignTradeXLineMaxValue*2.2 )
        self.ax3.set_ylim(ymin = ax3ymin, ymax = self.ForeignTradeYLineMaxValue*1.5)
        
        if ax3ymin < 0:
            self.ax3.spines['bottom'].set_position("zero")
        else:
            self.ax3.spines['bottom'].set_position(("data", ax3ymin))

    def EndoVarandParameters(self):

        [   X1_base, L1_base, K1_base, I11_base, I21_base, I31_base, Ir1_base, Ib1_base, Z1_base, E1_base, D1_base, Q1_base, M1_base, 
            X2_base, L2_base, K2_base, I12_base, I22_base, I32_base, Ir2_base, Ib2_base, Z2_base, E2_base, D2_base, Q2_base, M2_base, 
            X3_base, L3_base, K3_base, I13_base, I23_base, I33_base, Ir3_base, Ib3_base, Z3_base, E3_base, D3_base, Q3_base, M3_base, 
            Xr_base, Lr_base, Kr_base, COr_base, MCOr_base, DCOr_base, XCOr_base, I1r_base, I2r_base, I3r_base,  Irr_base, Ibr_base, Zr_base, Er_base, Dr_base, Qr_base, Mr_base, 
            Xb_base, Lb_base, Kb_base,  NGb_base, MNGb_base, DNGb_base, XNGb_base, I1b_base, I2b_base, I3b_base,  Irb_base,  Ibb_base, Zb_base, Eb_base, Db_base, 
            C1_base, C2_base, C3_base,  Cr_base, Cb_base, Y_base, self.Yd_base, 
            TPAO1_base, TPAO2_base, TPAO3_base,  OIL_INCOME_base, 
            G1_base, G2_base, G3_base,  T_base, Td_base, Tz_base, Tva_base, Tz1_base, Tz2_base, Tz3_base, Tzr_base, Tzb_base, Tva1_base, Tva2_base, Tva3_base,  Tvar_base, Tvab_base, 
            INV1_base, INV2_base, INV3_base, S_base, Sp_base, Sg_base,
            px1_base, px2_base, px3_base, pxr_base, pxb_base, 
            pz1_base, pz2_base, pz3_base, pzr_base, pzb_base, 
            pe1_base, pe2_base, pe3_base, per_base, peb_base, 
            pd1_base, pd2_base, pd3_base, pdr_base, pdb_base, 
            pq1_base, pq2_base, pq3_base, pqr_base, 
            pm1_base, pm2_base, pm3_base, pmr_base, 
            pmco_base, pdco_base, pco_base, pxco_base, 
            pmng_base, pdng_base, png_base, pxng_base, 
            Sf_base, self.r_base ] = self.result1.x

        [   X1_cfrun, L1_cfrun, K1_cfrun, I11_cfrun, I21_cfrun, I31_cfrun, Ir1_cfrun, Ib1_cfrun, Z1_cfrun, E1_cfrun, D1_cfrun, Q1_cfrun, M1_cfrun, 
            X2_cfrun, L2_cfrun, K2_cfrun, I12_cfrun, I22_cfrun, I32_cfrun, Ir2_cfrun, Ib2_cfrun, Z2_cfrun, E2_cfrun, D2_cfrun, Q2_cfrun, M2_cfrun, 
            X3_cfrun, L3_cfrun, K3_cfrun, I13_cfrun, I23_cfrun, I33_cfrun, Ir3_cfrun, Ib3_cfrun, Z3_cfrun, E3_cfrun, D3_cfrun, Q3_cfrun, M3_cfrun, 
            Xr_cfrun, Lr_cfrun, Kr_cfrun, COr_cfrun, MCOr_cfrun, DCOr_cfrun, XCOr_cfrun, I1r_cfrun, I2r_cfrun, I3r_cfrun,  Irr_cfrun, Ibr_cfrun, Zr_cfrun, Er_cfrun, Dr_cfrun, Qr_cfrun, Mr_cfrun, 
            Xb_cfrun, Lb_cfrun, Kb_cfrun,  NGb_cfrun, MNGb_cfrun, DNGb_cfrun, XNGb_cfrun, I1b_cfrun, I2b_cfrun, I3b_cfrun,  Irb_cfrun,  Ibb_cfrun, Zb_cfrun, Eb_cfrun, Db_cfrun, 
            C1_cfrun, C2_cfrun, C3_cfrun,  Cr_cfrun, Cb_cfrun, Y_cfrun, self.Yd_cfrun, 
            TPAO1_cfrun, TPAO2_cfrun, TPAO3_cfrun,  OIL_INCOME_cfrun, 
            G1_cfrun, G2_cfrun, G3_cfrun,  T_cfrun, Td_cfrun, Tz_cfrun, Tva_cfrun, Tz1_cfrun, Tz2_cfrun, Tz3_cfrun, Tzr_cfrun, Tzb_cfrun, Tva1_cfrun, Tva2_cfrun, Tva3_cfrun,  Tvar_cfrun, Tvab_cfrun, 
            INV1_cfrun, INV2_cfrun, INV3_cfrun, S_cfrun, Sp_cfrun, Sg_cfrun,
            px1_cfrun, px2_cfrun, px3_cfrun, pxr_cfrun, pxb_cfrun, 
            pz1_cfrun, pz2_cfrun, pz3_cfrun, pzr_cfrun, pzb_cfrun, 
            pe1_cfrun, pe2_cfrun, pe3_cfrun, per_cfrun, peb_cfrun, 
            pd1_cfrun, pd2_cfrun, pd3_cfrun, pdr_cfrun, pdb_cfrun, 
            pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun, 
            pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun, 
            pmco_cfrun, pdco_cfrun, pco_cfrun, pxco_cfrun, 
            pmng_cfrun, pdng_cfrun, png_cfrun, pxng_cfrun, 
            Sf_cfrun, self.r_cfrun ] = self.result2.x
        

        [td_base, tva1_base, tva2_base, tva3_base, tvar_base, tvab_base, tz1_base, tz2_base, tz3_base, tzr_base, tzb_base,
        alpha1_base, alpha2_base, alpha3_base, alphar_base,alphab_base, delta1_base, delta2_base,
        delta3_base, deltar_base, deltab_base, beta1_base, beta2_base, beta3_base,betar_base, betab_base, A1_base, A2_base, A3_base, 
        Ar_base, Ab_base, Axco_base, alphaxco_base, xr_base, co_base, Axng_base, alphaxng_base, xb_base, ng_base, 
        a11_base, a21_base, a31_base, ar1_base, ab1_base, a12_base, a22_base, a32_base, ar2_base, ab2_base, a13_base, a23_base,
        a33_base, ar3_base, ab3_base, a1r_base, a2r_base, a3r_base, arr_base, abr_base, a1b_base, a2b_base, a3b_base, arb_base, 
        abb_base, x1_base, x2_base, x3_base, xcor_base, xngb_base, rho1_base, rho2_base, rho3_base, rhor_base, rhob_base, eta1_base, 
        eta2_base, eta3_base, etar_base, etaco_base, etang_base, e1_base, e2_base, e3_base, er_base, eb_base, dt1_base, dt2_base, dt3_base, 
        dtr_base, dtb_base, theta1_base, theta2_base, theta3_base, thetar_base, thetab_base, m1_base, m2_base, m3_base, mr_base, mcor_base, 
        mngb_base, m1_base, m2_base, m3_base, mr_base, mcor_base, mngb_base, da1_base, da2_base, da3_base, dar_base, dcor_base, dngb_base, 
        lambda1_base, lambda2_base, lambda3_base, lambdar_base, lambdaco_base, lambdang_base, c1_base, c2_base, c3_base, cr_base, cb_base, 
        mu1_base, mu2_base, mu3_base, g1_base, g2_base, g3_base, inv1_base, inv2_base, inv3_base, sp_base, sg_base] = self.m1.model_parameters()

        [td_cfrun, tva1_cfrun, tva2_cfrun, tva3_cfrun, tvar_cfrun, tvab_cfrun, tz1_cfrun, tz2_cfrun, tz3_cfrun, tzr_cfrun, tzb_cfrun,
        alpha1_cfrun, alpha2_cfrun, alpha3_cfrun, alphar_cfrun,alphab_cfrun, delta1_cfrun, delta2_cfrun,
        delta3_cfrun, deltar_cfrun, deltab_cfrun, beta1_cfrun, beta2_cfrun, beta3_cfrun,betar_cfrun, betab_cfrun, A1_cfrun, A2_cfrun, A3_cfrun, 
        Ar_cfrun, Ab_cfrun, Axco_cfrun, alphaxco_cfrun, xr_cfrun, co_cfrun, Axng_cfrun, alphaxng_cfrun, xb_cfrun, ng_cfrun, 
        a11_cfrun, a21_cfrun, a31_cfrun, ar1_cfrun, ab1_cfrun, a12_cfrun, a22_cfrun, a32_cfrun, ar2_cfrun, ab2_cfrun, a13_cfrun, a23_cfrun,
        a33_cfrun, ar3_cfrun, ab3_cfrun, a1r_cfrun, a2r_cfrun, a3r_cfrun, arr_cfrun, abr_cfrun, a1b_cfrun, a2b_cfrun, a3b_cfrun, arb_cfrun, 
        abb_cfrun, x1_cfrun, x2_cfrun, x3_cfrun, xcor_cfrun, xngb_cfrun, rho1_cfrun, rho2_cfrun, rho3_cfrun, rhor_cfrun, rhob_cfrun, eta1_cfrun, 
        eta2_cfrun, eta3_cfrun, etar_cfrun, etaco_cfrun, etang_cfrun, e1_cfrun, e2_cfrun, e3_cfrun, er_cfrun, eb_cfrun, dt1_cfrun, dt2_cfrun, dt3_cfrun, 
        dtr_cfrun, dtb_cfrun, theta1_cfrun, theta2_cfrun, theta3_cfrun, thetar_cfrun, thetab_cfrun, m1_cfrun, m2_cfrun, m3_cfrun, mr_cfrun, mcor_cfrun, 
        mngb_cfrun, m1_cfrun, m2_cfrun, m3_cfrun, mr_cfrun, mcor_cfrun, mngb_cfrun, da1_cfrun, da2_cfrun, da3_cfrun, dar_cfrun, dcor_cfrun, dngb_cfrun, 
        lambda1_cfrun, lambda2_cfrun, lambda3_cfrun, lambdar_cfrun, lambdaco_cfrun, lambdang_cfrun, c1_cfrun, c2_cfrun, c3_cfrun, cr_cfrun, cb_cfrun, 
        mu1_cfrun, mu2_cfrun, mu3_cfrun, g1_cfrun, g2_cfrun, g3_cfrun, inv1_cfrun, inv2_cfrun, inv3_cfrun, sp_cfrun, sg_cfrun] = self.m2.model_parameters()



        self.D_base      = [D1_base, D2_base, D3_base, Dr_base]
        self.Q_base      = [Q1_base, Q2_base, Q3_base, Qr_base]
        self.M_base      = [M1_base, M2_base, M3_base, Mr_base]
        self.lambda_base = [lambda1_base, lambda2_base, lambda3_base, lambdar_base]
        self.eta_base    = [eta1_base, eta2_base, eta3_base, etar_base]
        self.m_base      = [m1_base, m2_base, m3_base, mr_base]
        self.da_base     = [da1_base, da2_base, da3_base, dar_base]
        self.pq_base     = [pq1_base, pq2_base, pq3_base, pqr_base]
        self.pd_base     = [pd1_base, pd2_base, pd3_base, pdr_base]
        self.pm_base     = [pm1_base, pm2_base, pm3_base, pmr_base]
        self.Tva_base     = [Tva1_base, Tva2_base, Tva3_base, Tvar_base]
        self.Tz_base     = [Tz1_base, Tz2_base, Tz3_base, Tzr_base]
        
        self.Z_base      = [Z1_base, Z2_base, Z3_base, Zr_base]
        self.E_base      = [E1_base, E2_base, E3_base, Er_base]
        self.theta_base  = [theta1_base, theta2_base, theta3_base, thetar_base]
        self.rho_base    = [rho1_base, rho2_base, rho3_base, rhor_base]
        self.e_base      = [e1_base, e2_base, e3_base, er_base]
        self.dt_base     = [dt1_base, dt2_base, dt3_base, dtr_base]
        self.tz_base     = [tz1_base, tz2_base, tz3_base, tzr_base]
        self.tva_base    = [tva1_base, tva2_base, tva3_base, tvar_base]
        self.pz_base     = [pz1_base, pz2_base, pz3_base, pzr_base]
        self.pe_base     = [pe1_base, pe2_base, pe3_base, per_base]

        self.D_cfrun      = [D1_cfrun, D2_cfrun, D3_cfrun, Dr_cfrun]
        self.Q_cfrun      = [Q1_cfrun, Q2_cfrun, Q3_cfrun, Qr_cfrun]
        self.M_cfrun      = [M1_cfrun, M2_cfrun, M3_cfrun, Mr_cfrun]
        self.lambda_cfrun = [lambda1_cfrun, lambda2_cfrun, lambda3_cfrun, lambdar_cfrun]
        self.eta_cfrun    = [eta1_cfrun, eta2_cfrun, eta3_cfrun, etar_cfrun]
        self.m_cfrun      = [m1_cfrun, m2_cfrun, m3_cfrun, mr_cfrun]
        self.da_cfrun     = [da1_cfrun, da2_cfrun, da3_cfrun, dar_cfrun]
        self.pq_cfrun     = [pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun]
        self.pd_cfrun     = [pd1_cfrun, pd2_cfrun, pd3_cfrun, pdr_cfrun]
        self.pm_cfrun     = [pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun]
        self.Tva_cfrun     = [Tva1_cfrun, Tva2_cfrun, Tva3_cfrun, Tvar_cfrun]
        self.Tz_cfrun     = [Tz1_cfrun, Tz2_cfrun, Tz3_cfrun, Tzr_cfrun]
        
        self.Z_cfrun      = [Z1_cfrun, Z2_cfrun, Z3_cfrun, Zr_cfrun]
        self.E_cfrun      = [E1_cfrun, E2_cfrun, E3_cfrun, Er_cfrun]
        self.theta_cfrun  = [theta1_cfrun, theta2_cfrun, theta3_cfrun, thetar_cfrun]
        self.rho_cfrun    = [rho1_cfrun, rho2_cfrun, rho3_cfrun, rhor_cfrun]
        self.e_cfrun      = [e1_cfrun, e2_cfrun, e3_cfrun, er_cfrun]
        self.dt_cfrun     = [dt1_cfrun, dt2_cfrun, dt3_cfrun, dtr_cfrun]
        self.tz_cfrun     = [tz1_cfrun, tz2_cfrun, tz3_cfrun, tzr_cfrun]
        self.tva_cfrun    = [tva1_cfrun, tva2_cfrun, tva3_cfrun, tvar_cfrun]
        self.pz_cfrun     = [pz1_cfrun, pz2_cfrun, pz3_cfrun, pzr_cfrun]
        self.pe_cfrun     = [pe1_cfrun, pe2_cfrun, pe3_cfrun, per_cfrun]

        self.Sf_base  = [M1_base*pm1_base - E1_base*pe1_base, M2_base*pm2_base - E2_base*pe2_base,
                          M3_base*pm3_base - E3_base*pe3_base, Mr_base*pmr_base - Er_base*per_base]
        self.Sf_cfrun = [M1_cfrun*pm1_cfrun - E1_cfrun*pe1_cfrun, M2_cfrun*pm2_cfrun - E2_cfrun*pe2_cfrun, 
                         M3_cfrun*pm3_cfrun - E3_cfrun*pe3_cfrun, Mr_cfrun*pmr_cfrun - Er_cfrun*per_cfrun]
        

        self.L_base     = [L1_base, L2_base, L3_base, Lr_base, Lb_base]
        self.K_base     = [K1_base, K2_base, K3_base, Kr_base, Kb_base]
        self.X_base     = [X1_base, X2_base, X3_base, Xr_base, Xb_base]
        self.A_base     = [A1_base, A2_base, A3_base, Ar_base, Ab_base]
        self.alpha_base = [alpha1_base, alpha2_base, alpha3_base, alphar_base, alphab_base]
        self.delta_base = [delta1_base, delta2_base, delta3_base, deltar_base, deltab_base]
        self.beta_base  = [beta1_base, beta2_base, beta3_base, betar_base, betab_base]
        self.px_base    = [px1_base, px2_base, px3_base, pxr_base, pxb_base]

        self.L_cfrun     = [L1_cfrun, L2_cfrun, L3_cfrun, Lr_cfrun, Lb_cfrun]
        self.K_cfrun     = [K1_cfrun, K2_cfrun, K3_cfrun, Kr_cfrun, Kb_cfrun]
        self.X_cfrun     = [X1_cfrun, X2_cfrun, X3_cfrun, Xr_cfrun, Xb_cfrun]
        self.A_cfrun     = [A1_cfrun, A2_cfrun, A3_cfrun, Ar_cfrun, Ab_cfrun]
        self.alpha_cfrun = [alpha1_cfrun, alpha2_cfrun, alpha3_cfrun, alphar_cfrun, alphab_cfrun]
        self.delta_cfrun = [delta1_cfrun, delta2_cfrun, delta3_cfrun, deltar_cfrun, deltab_cfrun]
        self.beta_cfrun  = [beta1_cfrun, beta2_cfrun, beta3_cfrun, betar_cfrun, betab_cfrun]
        self.px_cfrun    = [px1_cfrun, px2_cfrun, px3_cfrun, pxr_cfrun, pxb_cfrun]


        self.C_base = [C1_base, C2_base, C3_base, Cr_base, Cb_base]
        self.G_base = [G1_base, G2_base, G3_base]
        self.INV_base = [INV1_base, INV2_base, INV3_base]
        self.TPAO_base = [TPAO1_base, TPAO2_base, TPAO3_base]
        self.pq_base = [pq1_base, pq2_base, pq3_base, pqr_base]

        self.C_cfrun = [C1_cfrun, C2_cfrun, C3_cfrun, Cr_cfrun, Cb_cfrun]
        self.G_cfrun = [G1_cfrun, G2_cfrun, G3_cfrun]
        self.INV_cfrun = [INV1_cfrun, INV2_cfrun, INV3_cfrun]
        self.TPAO_cfrun = [TPAO1_cfrun, TPAO2_cfrun, TPAO3_cfrun]

        self.pq_cfrun = [pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun]

        self.I1_base = [I11_base, I12_base, I13_base]
        self.I2_base = [I21_base, I22_base, I23_base]
        self.I3_base = [I31_base, I32_base, I33_base]


        self.I1_cfrun = [I11_cfrun, I21_cfrun, I31_cfrun]
        self.I2_cfrun = [I12_cfrun, I22_cfrun, I32_cfrun]
        self.I3_cfrun = [I13_cfrun, I23_cfrun, I33_cfrun]

        # Macro Variables
        # ---------------
        self.GDP_base_total = Y_base + OIL_INCOME_base + Tva_base + Tz_base
        self.CES_base_total  = C1_base*pq1_base + C2_base*pq2_base + C3_base*pq3_base + Cr_base*pqr_base + Cb_base*pdb_base
        self.TPAO_base_total = TPAO1_base*pq1_base + TPAO2_base*pq2_base + TPAO3_base*pq3_base
        self.G_base_total   = G1_base*pq1_base + G2_base*pq2_base + G3_base*pq3_base
        self.INV_base_total = INV1_base*pq1_base + INV2_base*pq2_base + INV3_base*pq3_base
        self.E_base_total   = E1_base * pe1_base + E2_base*pe2_base + E3_base*pe3_base + Er_base*per_base + Eb_base*peb_base + self.m1.epsilon * self.m1.E_Energy
        self.M_base_total   = M1_base*pm1_base + M2_base*pm2_base + M3_base*pm3_base + MCOr_base*pmco_base + Mr_base*pmr_base + MNGb_base*pmng_base
        self.Total_base_Consumption =  self.CES_base_total +  self.TPAO_base_total + self.G_base_total  + self.INV_base_total + self.E_base_total - self.M_base_total 
        self.NE_base_total  = self.E_base_total - self.M_base_total
        self.Tva_base_total = Tva1_base + Tva2_base + Tva3_base + Tvar_base + Tvab_base
        self.Tz_base_total  = Tz1_base + Tz2_base + Tz3_base + Tzr_base + Tzb_base
        
        self.GDP_cfrun_total = Y_cfrun + OIL_INCOME_cfrun + Tva_cfrun + Tz_cfrun
        self.CES_cfrun_total = C1_cfrun*pq1_cfrun + C2_cfrun*pq2_cfrun + C3_cfrun*pq3_cfrun + Cr_base*pqr_base + Cb_base*pdb_base
        self.TPAO_cfrun_total = TPAO1_cfrun*pq1_cfrun + TPAO2_cfrun*pq2_cfrun + TPAO3_cfrun*pq3_cfrun
        self.G_cfrun_total   = G1_cfrun*pq1_cfrun + G2_cfrun*pq2_cfrun + G3_cfrun*pq3_cfrun
        self.INV_cfrun_total = INV1_cfrun*pq1_cfrun + INV2_cfrun*pq2_cfrun + INV3_cfrun*pq3_cfrun
        self.E_cfrun_total   = E1_cfrun * pe1_cfrun + E2_cfrun*pe2_cfrun + E3_cfrun*pe3_cfrun + Er_cfrun*per_cfrun + Eb_cfrun*peb_cfrun + self.m2.epsilon * self.m2.E_Energy
        self.M_cfrun_total   = M1_cfrun*pm1_cfrun + M2_cfrun*pm2_cfrun + M3_cfrun*pm3_cfrun + MCOr_cfrun*pmco_cfrun + Mr_cfrun*pmr_cfrun + MNGb_cfrun*pmng_cfrun
        self.Total_cfrun_Consumption =  self.CES_cfrun_total +  self.TPAO_cfrun_total + self.G_cfrun_total  + self.INV_cfrun_total +  self.E_cfrun_total - self.M_cfrun_total 
        self.NE_cfrun_total  = self.E_cfrun_total - self.M_cfrun_total
        self.Tva_cfrun_total = Tva1_cfrun + Tva2_cfrun + Tva3_cfrun + Tvar_cfrun + Tvab_cfrun
        self.Tz_cfrun_total  = Tz1_cfrun + Tz2_cfrun + Tz3_cfrun + Tzr_cfrun + Tzb_cfrun


        self.index = ["GDP Gelir", "Özel Tüketim", "TPAO", "Kamu", "Yatırım", "İhracat", "İthalat", "Toplam Harcama", "Net İhracat", "Ürün Vergi", "Üretim Vergi"]
       
        self.base_values = [self.GDP_base_total,self.CES_base_total, self.TPAO_base_total, self.G_base_total, self.INV_base_total, self.E_base_total, self.M_base_total,  
                            self.Total_base_Consumption, self.NE_base_total, self.Tva_base_total, self.Tz_base_total ]
        
        self.cfrun_values = [self.GDP_cfrun_total,self.CES_cfrun_total, self.TPAO_cfrun_total, self.G_cfrun_total, self.INV_cfrun_total, self.E_cfrun_total, self.M_cfrun_total,  
                            self.Total_cfrun_Consumption, self.NE_cfrun_total, self.Tva_cfrun_total, self.Tz_cfrun_total ]
              

        self.arm_budget_fridges = {"Mb":self.pq_base[self.sec_code]*self.Q_base[self.sec_code]/(self.pm_base[self.sec_code]),
                                   "Mc":self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code]/(self.pm_cfrun[self.sec_code]),
                                   "Db":self.pq_base[self.sec_code]*self.Q_base[self.sec_code]/self.pd_base[self.sec_code], 
                                   "Dc":self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code]/self.pd_cfrun[self.sec_code]}
        
        self.trs_budget_fridges = {"Eb":(1+self.tz_base[self.sec_code] + self.tva_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pe_base[self.sec_code],
                                   "Ec":(1+self.tz_cfrun[self.sec_code] + self.tva_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] ,
                                   "Db":(1+self.tz_base[self.sec_code] + self.tva_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pd_base[self.sec_code], 
                                   "Dc":(1+self.tz_cfrun[self.sec_code] + self.tva_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pd_cfrun[self.sec_code]}
        
        self.ces_budget_fridges = {"Kb":self.px_base[self.sec_code]*self.X_base[self.sec_code] / self.r_base,
                                   "Kc":self.px_cfrun[self.sec_code]*self.X_cfrun[self.sec_code] / self.r_cfrun, 
                                   "Lb":self.px_base[self.sec_code]*self.X_base[self.sec_code],
                                   "Lc":self.px_cfrun[self.sec_code]*self.X_cfrun[self.sec_code]}
        
        self.arm_fridges =  {"Mb": ((self.Q_base[self.sec_code]/self.lambda_base[self.sec_code])**self.eta_base[self.sec_code]*(1/self.m_base[self.sec_code]))**(1/self.eta_base[self.sec_code]),
                             "Mc": ((self.Q_cfrun[self.sec_code]/self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code]*(1/self.m_cfrun[self.sec_code]))**(1/self.eta_cfrun[self.sec_code]),
                             "Db": ((self.Q_base[self.sec_code]/self.lambda_base[self.sec_code])**self.eta_base[self.sec_code]*1/self.da_base[self.sec_code])**(1/self.eta_base[self.sec_code]), 
                             "Dc": ((self.Q_cfrun[self.sec_code]/self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code]*(1/self.da_cfrun[self.sec_code]))**(1/self.eta_cfrun[self.sec_code])}
        
        self.trs_fridges =  {"Eb": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code]*1/self.e_base[self.sec_code])**(1/self.rho_base[self.sec_code]),
                             "Ec": ((self.Z_cfrun[self.sec_code]/ self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code]*1/self.e_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code]),
                             "Db": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code]*1/self.dt_base[self.sec_code])**(1/self.rho_base[self.sec_code]), 
                             "Dc": ((self.Z_cfrun[self.sec_code]/ self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code]*1/self.dt_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code])}
        
        self.ces_fridges =  {"Kb": ((self.X_base[self.sec_code]/self.A_base[self.sec_code])**self.alpha_base[self.sec_code]*(1/self.delta_base[self.sec_code]))**(1/self.alpha_base[self.sec_code]),
                             "Kc": ((self.X_cfrun[self.sec_code]/self.A_cfrun[self.sec_code])**self.alpha_cfrun[self.sec_code]*(1/self.delta_cfrun[self.sec_code]))**(1/self.alpha_cfrun[self.sec_code]),
                             "Lb": ((self.X_base[self.sec_code]/self.A_base[self.sec_code])**self.alpha_base[self.sec_code]*1/self.beta_base[self.sec_code])**(1/self.alpha_base[self.sec_code]), 
                             "Lc": ((self.X_cfrun[self.sec_code]/self.A_cfrun[self.sec_code])**self.alpha_cfrun[self.sec_code]*1/self.beta_cfrun[self.sec_code])**(1/self.alpha_cfrun[self.sec_code])}

    
        self.HouseholdBaseIncome = Y_base
        self.HouseholdCfrunIncome = Y_cfrun
        self.GovBaseIncome = T_base
        self.GovCfrunIncome = T_cfrun
        self.SavingsBase = S_base
        self.SavingsCfrun = S_cfrun
        self.EIncomeBase =  OIL_INCOME_base
        self.EIncomeCfrun =  OIL_INCOME_cfrun

        self.PriceBase = [px1_base, px2_base, px3_base, pxr_base, pxb_base, pz1_base, pz2_base, pz3_base, pzr_base, pzb_base, pe1_base, pe2_base, pe3_base, per_base, peb_base, 
                          pd1_base, pd2_base, pd3_base, pdr_base, pdb_base, pq1_base, pq2_base, pq3_base, pqr_base, pm1_base, pm2_base, pm3_base, pmr_base, pmco_base,
                          pco_base, pxco_base, pmng_base,png_base, pxng_base, self.r_base ]
        
        self.PriceCfrun = [px1_cfrun, px2_cfrun, px3_cfrun, pxr_cfrun, pxb_cfrun, pz1_cfrun, pz2_cfrun, pz3_cfrun, pzr_cfrun, pzb_cfrun, pe1_cfrun, pe2_cfrun, pe3_cfrun, per_cfrun, peb_cfrun, 
                          pd1_cfrun, pd2_cfrun, pd3_cfrun, pdr_cfrun, pdb_cfrun, pq1_cfrun, pq2_cfrun, pq3_cfrun, pqr_cfrun, pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun, pmco_cfrun,
                          pco_cfrun, pxco_cfrun, pmng_cfrun,png_cfrun, pxng_cfrun, self.r_cfrun ]
        

        cfrunQuantityValues = {
        "agr"  : [I11_cfrun, I12_cfrun, I13_cfrun, I1r_cfrun, I1b_cfrun, 0,0,0,0,0,0,0,0,0, C1_cfrun, TPAO1_cfrun, G1_cfrun, INV1_cfrun, 0, E1_cfrun],
        "ser"  : [I21_cfrun, I22_cfrun, I23_cfrun, I2r_cfrun, I2b_cfrun, 0,0,0,0,0,0,0,0,0, C2_cfrun, TPAO2_cfrun, G2_cfrun, INV2_cfrun, 0, E2_cfrun],
        "ind"  : [I31_cfrun, I32_cfrun, I33_cfrun, I3r_cfrun, I3b_cfrun, 0,0,0,0,0,0,0,0,0, C3_cfrun, TPAO3_cfrun, G3_cfrun, INV3_cfrun, 0, E3_cfrun],
        "raf"  : [Ir1_cfrun, Ir2_cfrun, Ir3_cfrun, Irr_cfrun, Irb_cfrun, 0,0,0,0,0,0,0,0,0, Cr_cfrun, 0,0,0, 0, Er_cfrun],
        "bts"  : [Ib1_cfrun, Ib2_cfrun, Ib3_cfrun, Ibr_cfrun, Ibb_cfrun, 0,0,0,0,0,0,0,0,0, Cb_cfrun, 0,0,0, 0, Eb_cfrun],
        "lab"  : [L1_cfrun, L2_cfrun, L3_cfrun, Lr_cfrun, Lb_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "cap"  : [K1_cfrun, K2_cfrun, K3_cfrun, Kr_cfrun, Kb_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "mco"  : [0,0,0, MCOr_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dco"  : [0,0,0, DCOr_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "mng"  : [0,0,0,0, MNGb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dng"  : [0,0,0,0, DNGb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "dtax" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,Td_cfrun,0,0,0, 0,0],
        "gtax" : [Tva1_cfrun, Tva2_cfrun, Tva3_cfrun, Tvar_cfrun, Tvab_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "ptax" : [Tz1_cfrun, Tz2_cfrun, Tz3_cfrun, Tzr_cfrun, Tzb_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "hh"   : [0,0,0,0,0, self.m2.Lbar, self.m2.Kbar,0,0,0,0,0,0,0,0,0,0,0, 0,0],
        "TPAO" : [0,0,0,0,0,0,0,0,DCOr_cfrun, 0, DNGb_cfrun, 0,0,0,0,0,0,0, self.m2.E_Energy,0],
        "gov"  : [0,0,0,0,0,0,0,0,0,0,0,Td_cfrun, Tva_cfrun, Tz_cfrun, 0,0,0,0, 0,0],
        "sav"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,Sp_cfrun, 0, Sg_cfrun, 0, 0, Sf_cfrun],
        "eexp"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0,self.m2.E_Energy],
        "imp"  : [M1_cfrun, M2_cfrun, M3_cfrun, Mr_cfrun,0,0,0,MCOr_cfrun, 0, MNGb_cfrun, 0,0,0,0,0,0,0,0, 0,0]}

        self.CFRunQuantitiySAM = pd.DataFrame(index = SAM.index[0:-1], columns = SAM.columns[0:-1])
        for key in cfrunQuantityValues.keys():
            self.CFRunQuantitiySAM.loc[key] = cfrunQuantityValues[key]

        self.CFRunQuantitiySAM.loc["total"] = self.CFRunQuantitiySAM.sum()
        self.CFRunQuantitiySAM["total"] = self.CFRunQuantitiySAM.sum(axis = 1)
        



        cfrunPriceValues = {
        "agr"  : [pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun, 0,0,0,0,0,0,0,0,0, pq1_cfrun, pq1_cfrun, pq1_cfrun, pq1_cfrun,0, pe1_cfrun],
        "ser"  : [pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun, 0,0,0,0,0,0,0,0,0, pq2_cfrun, pq2_cfrun, pq2_cfrun, pq2_cfrun,0, pe2_cfrun],
        "ind"  : [pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun, 0,0,0,0,0,0,0,0,0, pq3_cfrun, pq3_cfrun, pq3_cfrun, pq3_cfrun,0, pe3_cfrun],
        "raf"  : [pqr_cfrun, pqr_cfrun, pqr_cfrun, pqr_cfrun, pqr_cfrun, 0,0,0,0,0,0,0,0,0, pqr_cfrun, 0,0,0,0, per_cfrun],
        "bts"  : [pdb_cfrun, pdb_cfrun, pdb_cfrun, pdb_cfrun, pdb_cfrun, 0,0,0,0,0,0,0,0,0, pdb_cfrun, 0,0,0,0, peb_cfrun],
        "lab"  : [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "cap"  : [self.r_cfrun, self.r_cfrun, self.r_cfrun, self.r_cfrun, self.r_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "mco"  : [0,0,0, pmco_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dco"  : [0,0,0, pdco_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "mng"  : [0,0,0,0, pmng_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dng"  : [0,0,0,0, pdng_cfrun, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dtax" : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        "gtax" : [1, 1, 1, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "ptax" : [1, 1, 1, 1, 1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "hh"   : [0,0,0,0,0, 1,self.r_cfrun,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "TPAO" : [0,0,0,0,0,0,0,0,pdco_cfrun, 0, pdng_cfrun, 0,0,0,0,0,0,0,self.m2.epsilon, 0],
        "gov"  : [0,0,0,0,0,0,0,0,0,0,0,1, 1, 1, 0,0,0,0,0,0],
        "sav"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 0, 1, 0,0, self.m2.epsilon],
        "eexp"  : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, self.m2.epsilon],
        "imp"  : [pm1_cfrun, pm2_cfrun, pm3_cfrun, pmr_cfrun,0,0,0,pmco_cfrun, 0, pmng_cfrun, 0,0,0,0,0,0,0,0,0,0]}


        cfrunDatabaseValues = {
        "agr"  : [],
        "ser"  : [],
        "ind"  : [],
        "raf"  : [],
        "bts"  : [],
        "lab"  : [],
        "cap"  : [],
        "mco"  : [],
        "dco"  : [],
        "mng"  : [],
        "dng"  : [],
        "dtax" : [],
        "gtax" : [],
        "ptax" : [],
        "hh"   : [],
        "TPAO" : [],
        "gov"  : [],
        "sav"  : [],
        "eexp"  : [],
        "imp"  : []}


        for i in cfrunQuantityValues.keys():
            for j in range(len(cfrunQuantityValues[i])):

                quantity = cfrunQuantityValues[i][j]
                price    = cfrunPriceValues[i][j]
                value = quantity*price

                cfrunDatabaseValues[i].append(value)


        cfrunDatabase = pd.DataFrame(index = SAM.index, columns = SAM.columns[0:-1])

        for i in cfrunDatabaseValues.keys():
            cfrunDatabase.loc[i] = cfrunDatabaseValues[i]

        cfrunDatabase.loc["total"] = cfrunDatabase.sum()
        cfrunDatabase["total"] = cfrunDatabase.sum(axis = 1)

        self.CFSHM = cfrunDatabase
            
        cfrunDatabase.to_excel("SHMCFrun_.xlsx")


        # print(cfrunDatabase)
        # print(cfrunDatabase.loc["total"] - cfrunDatabase["total"])

    def ArmingtonFunction(self, X):

        return {"Base": ((self.Q_base[self.sec_code] / self.lambda_base[self.sec_code])**self.eta_base[self.sec_code] * 1/self.m_base[self.sec_code] - self.da_base[self.sec_code]/self.m_base[self.sec_code] * X **self.eta_base[self.sec_code])**(1/self.eta_base[self.sec_code]),
                "CFRun":((self.Q_cfrun[self.sec_code] / self.lambda_cfrun[self.sec_code])**self.eta_cfrun[self.sec_code] * 1/self.m_cfrun[self.sec_code] - self.da_cfrun[self.sec_code]/self.m_cfrun[self.sec_code] * X**self.eta_cfrun[self.sec_code])**(1/self.eta_cfrun[self.sec_code]) }
    
    def ArmingtonBudget(self, X):

        return {"Base" : self.pq_base[self.sec_code]*self.Q_base[self.sec_code] / (self.pm_base[self.sec_code] ) - self.pd_base[self.sec_code] / self.pm_base[self.sec_code] * X,
                "CFRun": self.pq_cfrun[self.sec_code]*self.Q_cfrun[self.sec_code] / (self.pm_cfrun[self.sec_code] ) - self.pd_cfrun[self.sec_code] / self.pm_cfrun[self.sec_code]* X }
    
    def TransformationFunction(self, X):

        return  {"Base": ((self.Z_base[self.sec_code] / self.theta_base[self.sec_code])**self.rho_base[self.sec_code] * (1/self.e_base[self.sec_code]) - (self.dt_base[self.sec_code]/self.e_base[self.sec_code] )* X**self.rho_base[self.sec_code])**(1/self.rho_base[self.sec_code]),
                 "CFRun":((self.Z_cfrun[self.sec_code] / self.theta_cfrun[self.sec_code])**self.rho_cfrun[self.sec_code] * (1/self.e_cfrun[self.sec_code]) - self.dt_cfrun[self.sec_code]/self.e_cfrun[self.sec_code]* X**self.rho_cfrun[self.sec_code])**(1/self.rho_cfrun[self.sec_code]) }  
    
    def TransformationBudget(self, X):

        return {"Base": (1+self.tva_base[self.sec_code]+self.tz_base[self.sec_code])*self.pz_base[self.sec_code]*self.Z_base[self.sec_code] / self.pe_base[self.sec_code] - self.pd_base[self.sec_code] / self.pe_base[self.sec_code] * X,
                "CFRun":(1+self.tva_cfrun[self.sec_code]+self.tz_cfrun[self.sec_code])*self.pz_cfrun[self.sec_code]*self.Z_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] - self.pd_cfrun[self.sec_code] / self.pe_cfrun[self.sec_code] * X}

    def ForeignTrade(self,X):

        return {"Base" : self.Sf_base[self.sec_code] / self.pm_base[self.sec_code]+ self.pe_base[self.sec_code]/self.pm_base[self.sec_code] * X ,
                "CFRun": self.Sf_cfrun[self.sec_code] / self.pm_cfrun[self.sec_code]  + self.pe_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code] * X}
    
    def CES(self, X, sec_code):

         return {"Base" : ((self.X_base[sec_code] / self.A_base[sec_code])**self.alpha_base[sec_code] * 1/self.beta_base[sec_code] - self.delta_base[sec_code]/self.beta_base[sec_code] * X **self.alpha_base[sec_code])**(1/self.alpha_base[sec_code]),
                 "CFRun": ((self.X_cfrun[sec_code] / self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code] * 1/self.beta_cfrun[sec_code] - self.delta_cfrun[sec_code]/self.beta_cfrun[sec_code] * X **self.alpha_cfrun[sec_code])**(1/self.alpha_cfrun[sec_code])}

    def CESFridges(self, sec_code):

        return {"ces_budget_fridges" : {"Kb":self.px_base[sec_code]*self.X_base[sec_code] / self.r_base,
                                        "Kc":self.px_cfrun[sec_code]*self.X_cfrun[sec_code] / self.r_cfrun, 
                                        "Lb":self.px_base[sec_code]*self.X_base[sec_code],
                                        "Lc":self.px_cfrun[sec_code]*self.X_cfrun[sec_code]},

                "ces_fridges" :  {"Kb": ((self.X_base[sec_code]/self.A_base[sec_code])**self.alpha_base[sec_code]*(1/self.delta_base[sec_code]))**(1/self.alpha_base[sec_code]),
                                  "Kc": ((self.X_cfrun[sec_code]/self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code]*(1/self.delta_cfrun[sec_code]))**(1/self.alpha_cfrun[sec_code]),
                                  "Lb": ((self.X_base[sec_code]/self.A_base[sec_code])**self.alpha_base[sec_code]*1/self.beta_base[sec_code])**(1/self.alpha_base[sec_code]), 
                                  "Lc": ((self.X_cfrun[sec_code]/self.A_cfrun[sec_code])**self.alpha_cfrun[sec_code]*1/self.beta_cfrun[sec_code])**(1/self.alpha_cfrun[sec_code])}
  
        }
    
    def CESBudget(self, X, sec_code):

        return {"Base" : self.px_base[sec_code]*self.X_base[sec_code] / self.r_base - 1 / self.r_base * X,
                "CFRun": self.px_cfrun[sec_code]*self.X_cfrun[sec_code] / self.r_cfrun - 1 / self.r_cfrun * X}

    def ArmingtonValues(self):

        Db = self.arm_budget_fridges["Db"]
        Dc = self.arm_budget_fridges["Dc"]
        Mb = self.arm_budget_fridges["Mb"]
        Mc = self.arm_budget_fridges["Mc"]

        #-------------------------Base Run Equilibrium-------------------------------------

        self.arm_budget_1_x_base = np.linspace(0, Db, 100)
        self.arm_budget_1_y_base = self.ArmingtonBudget(self.arm_budget_1_x_base)["Base"]

        X = np.linspace(0, Db, 100)
        Y = self.ArmingtonFunction(X)["Base"]

        self.arm_1_x_base = []
        self.arm_1_y_base = []

        for i in range(len(Y)):
            if Y[i] > Mb or Y[i] <0:
                continue

            self.arm_1_x_base.append(X[i])
            self.arm_1_y_base.append(Y[i])

        #----------------------CounterFactual Eqquilibrium---------------------------------

        self.arm_budget_1_x_cfrun = np.linspace(0, Dc, 100)
        self.arm_budget_1_y_cfrun = self.ArmingtonBudget(self.arm_budget_1_x_cfrun)["CFRun"]
        

        X = np.linspace(0, Dc, 100)
        Y = self.ArmingtonFunction(X)["CFRun"]

        self.arm_1_x_cfrun= []
        self.arm_1_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Mc or Y[i] < 0:
                continue

            self.arm_1_x_cfrun.append(X[i])
            self.arm_1_y_cfrun.append(Y[i])

    def axes1(self):

        self.ArmingtonValues()
        #----------------Base Year Equilibrium----------------------------
        label_base_arm =  "Base Year\nQ: {:.2f}\nD: {:.2f}\nM: {:.2f}\npd: {:.2f}\npm: {:.2f}\npq: {:.2f}".format(self.Q_base[self.sec_code], 
                                                                                              self.D_base[self.sec_code], self.M_base[self.sec_code], 
                                                                                              self.pd_base[self.sec_code],
                                                                                              self.pm_base[self.sec_code], 
                                                                                              self.pq_base[self.sec_code])
        
        self.ax1.plot(self.arm_1_x_base, self.arm_1_y_base, "-k", linewidth = 3, label = label_base_arm)
        self.ax1.plot(self.arm_budget_1_x_base, self.arm_budget_1_y_base, "-k", linewidth = 3)
        self.ax1.plot([0, self.D_base[self.sec_code]], [self.M_base[self.sec_code], self.M_base[self.sec_code]], "--ok")
        self.ax1.plot([self.D_base[self.sec_code], self.D_base[self.sec_code]], [0, self.M_base[self.sec_code]], "--ok")

        #------------Counterfactual Equilibrium----------------------------
        label_cfrun_arm =  "CF Equilibrium\nQ: {:.2f}\nD: {:.2f}\nM: {:.2f}\npd: {:.2f}\npm: {:.2f}\npq: {:.2f}".format(self.Q_cfrun[self.sec_code], 
                                                                                              self.D_cfrun[self.sec_code], self.M_cfrun[self.sec_code], 
                                                                                              self.pd_cfrun[self.sec_code],
                                                                                              self.pm_cfrun[self.sec_code],
                                                                                              self.pq_cfrun[self.sec_code])
        
        self.ax1.plot(self.arm_1_x_cfrun, self.arm_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_arm)
        self.ax1.plot(self.arm_budget_1_x_cfrun, self.arm_budget_1_y_cfrun, "--k", linewidth = 1)
        self.ax1.plot([0, self.D_cfrun[self.sec_code]], [self.M_cfrun[self.sec_code], self.M_cfrun[self.sec_code]], "--ok")
        self.ax1.plot([self.D_cfrun[self.sec_code], self.D_cfrun[self.sec_code]], [0, self.M_cfrun[self.sec_code]], "--ok")

        #----------------------Axes-------------------------------------
        self.ax1.set_title("Armington Function", fontweight = "bold", fontsize = 13)
        self.ax1.set_xlabel("Domestic Demand",fontweight = "bold", fontsize = 11)
        self.ax1.set_ylabel("Import Demand",fontweight = "bold", fontsize = 11)
        self.ax1.spines["right"].set_visible(False)
        self.ax1.spines["top"].set_visible(False)
        self.ax1.legend(fontsize = 7, edgecolor = "black", ncol = 2, loc = 1)

    def TransformationValues(self):


        BaseBudgetMin = self.D_base[self.sec_code] - self.D_base[self.sec_code]/3
        BaseBudgetMax = self.D_base[self.sec_code] + self.D_base[self.sec_code]/3

        CfrunBudgetMin = self.D_cfrun[self.sec_code] - self.D_cfrun[self.sec_code]/3
        CfrunBudgetMax = self.D_cfrun[self.sec_code] + self.D_cfrun[self.sec_code]/3

        Db = self.trs_fridges["Db"]
        Dc = self.trs_fridges["Dc"]




        self.trs_budget_1_x_base = np.linspace(BaseBudgetMin, BaseBudgetMax , 100)
        self.trs_budget_1_y_base = self.TransformationBudget(self.trs_budget_1_x_base)["Base"]

        self.trs_1_x_base = np.linspace(0, Db, 100)
        self.trs_1_y_base = self.TransformationFunction(self.trs_1_x_base)["Base"]

        self.trs_budget_1_x_cfrun = np.linspace(CfrunBudgetMin, CfrunBudgetMax, 100)
        self.trs_budget_1_y_cfrun = self.TransformationBudget(self.trs_budget_1_x_cfrun)["CFRun"]

        self.trs_1_x_cfrun = np.linspace(0, Dc, 100)
        self.trs_1_y_cfrun = self.TransformationFunction(self.trs_1_x_cfrun)["CFRun"]
  
    def axes2(self):

        self.TransformationValues()
        #----------------Base Year Equilibrium----------------------------
        label_base_trs =  "Base Year\nZ: {:.2f}\nD: {:.2f}\nE: {:.2f}\npd: {:.2f}\npe: {:.2f}\npz: {:.2f}".format(self.Z_base[self.sec_code], 
                                                                                              self.D_base[self.sec_code], self.E_base[self.sec_code],
                                                                                              self.pd_base[self.sec_code],
                                                                                              self.pe_base[self.sec_code],
                                                                                              self.pz_base[self.sec_code])

        self.ax2.plot(self.trs_1_x_base, self.trs_1_y_base, "-k", linewidth = 3, label = label_base_trs)
        self.ax2.plot(self.trs_budget_1_x_base, self.trs_budget_1_y_base, "-k", linewidth = 3)

        self.ax2.plot([0, self.D_base[self.sec_code]], [self.E_base[self.sec_code], self.E_base[self.sec_code]], "--ok")
        self.ax2.plot([self.D_base[self.sec_code], self.D_base[self.sec_code]], [0, self.E_base[self.sec_code]], "--ok")

        #------------Counterfactual Equilibrium----------------------------
        label_cfrun_trs =  "CF Equilibrium\nZ: {:.2f}\nD: {:.2f}\nE: {:.2f}\npd: {:.2f}\npe: {:.2f}\npz: {:.2f}".format(self.Z_cfrun[self.sec_code], 
                                                                                              self.D_cfrun[self.sec_code], self.E_cfrun[self.sec_code],
                                                                                              self.pd_cfrun[self.sec_code],
                                                                                              self.pe_cfrun[self.sec_code], 
                                                                                              self.pz_cfrun[self.sec_code])
        

        self.ax2.plot(self.trs_1_x_cfrun, self.trs_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_trs)
        self.ax2.plot(self.trs_budget_1_x_cfrun, self.trs_budget_1_y_cfrun, "--k", linewidth = 1)
        self.ax2.plot([0, self.D_cfrun[self.sec_code]], [self.E_cfrun[self.sec_code], self.E_cfrun[self.sec_code]], "--ok")
        self.ax2.plot([self.D_cfrun[self.sec_code], self.D_cfrun[self.sec_code]], [0, self.E_cfrun[self.sec_code]], "--ok")


        #----------------------Axes-------------------------------------
        self.ax2.set_title("Transformation Function".format(self.sectors[self.sec_code]), fontweight = "bold", fontsize = 13)
        self.ax2.set_xlabel("Domestic Supply",fontweight = "bold", fontsize = 11)
        self.ax2.set_ylabel("Export Supply",fontweight = "bold", fontsize = 11)
        self.ax2.spines["right"].set_visible(False)
        self.ax2.spines["top"].set_visible(False)

        self.ax2.legend(fontsize = 7, edgecolor = "black", ncol = 2, loc = 1)

    def ForeignTradeValues(self):

        self.trade_line_1_x_base = np.linspace(0, self.ForeignTradeXLineMaxValue*2, 100)
        self.trade_line_1_y_base = self.ForeignTrade(self.trade_line_1_x_base)["Base"]

        self.trade_line_1_x_cfrun = np.linspace(0, self.ForeignTradeXLineMaxValue*2, 100)
        self.trade_line_1_y_cfrun = self.ForeignTrade(self.trade_line_1_x_cfrun)["CFRun"]
    
    def axes3(self):


        self.ForeignTradeValues()

        label_base_foreign =  "Base Year\nE: {:.2f}\nM: {:.2f}\nSf/pm: {:.2f}\npe/pm: {:.2f}".format(self.E_base[self.sec_code], self.M_base[self.sec_code],
                                                                                                    self.Sf_base[self.sec_code]/self.pm_base[self.sec_code],
                                                                                                    self.pe_base[self.sec_code]/self.pm_base[self.sec_code])
        
        label_cfrun_foreign =  "CF Equilibrium\nE: {:.2f}\nM: {:.2f}\nSf/pm: {:.2f}\npe/pm: {:.2f}".format(self.E_cfrun[self.sec_code], self.M_cfrun[self.sec_code], 
                                                                                                                   self.Sf_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code], 
                                                                                                                   self.pe_cfrun[self.sec_code]/self.pm_cfrun[self.sec_code])

        self.ax3.plot(self.trade_line_1_x_base, self.trade_line_1_y_base, "-k", linewidth = 3, label = label_base_foreign)


        self.ax3.plot(self.trade_line_1_x_cfrun, self.trade_line_1_y_cfrun, "--k", linewidth = 1, label = label_cfrun_foreign)

        self.ax3.plot([0, self.E_base[self.sec_code]], [self.M_base[self.sec_code], self.M_base[self.sec_code]], "--ok")
        self.ax3.plot([self.E_base[self.sec_code], self.E_base[self.sec_code]], [0, self.M_base[self.sec_code]], "--ok")

        self.ax3.plot([0, self.E_cfrun[self.sec_code]], [self.M_cfrun[self.sec_code], self.M_cfrun[self.sec_code]], "--ok")
        self.ax3.plot([self.E_cfrun[self.sec_code], self.E_cfrun[self.sec_code]], [0, self.M_cfrun[self.sec_code]], "--ok")

        #----------------------Axes-------------------------------------
        self.ax3.set_title("Foreign Trade", fontweight = "bold", fontsize = 13)
        self.ax3.set_xlabel("Export",fontweight = "bold", fontsize = 11)
        self.ax3.set_ylabel("Import",fontweight = "bold", fontsize = 11)
        self.ax3.spines["right"].set_visible(False)
        self.ax3.spines["top"].set_visible(False)

        self.ax3.legend(fontsize = 7, edgecolor = "black", ncol = 2, loc = 4)

    def ArmingtonCETPlot(self):

        fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1,3, figsize = (15, 4))

        plt.tight_layout()
        self.AxisLimits()
        self.axes1()
        self.axes2()
        self.axes3()
        plt.show()

    def CesValues(self):

        Lb0 = self.CESFridges(0)["ces_budget_fridges"]["Lb"]
        Lc0 = self.CESFridges(0)["ces_budget_fridges"]["Lc"]
        Kb0 = self.CESFridges(0)["ces_budget_fridges"]["Kb"]
        Kc0 = self.CESFridges(0)["ces_budget_fridges"]["Kc"]

        Lb1 = self.CESFridges(1)["ces_budget_fridges"]["Lb"]
        Lc1 = self.CESFridges(1)["ces_budget_fridges"]["Lc"]
        Kb1 = self.CESFridges(1)["ces_budget_fridges"]["Kb"]
        Kc1 = self.CESFridges(1)["ces_budget_fridges"]["Kc"]

        Lb2 = self.CESFridges(2)["ces_budget_fridges"]["Lb"]
        Lc2 = self.CESFridges(2)["ces_budget_fridges"]["Lc"]
        Kb2 = self.CESFridges(2)["ces_budget_fridges"]["Kb"]
        Kc2 = self.CESFridges(2)["ces_budget_fridges"]["Kc"]

        Lbr = self.CESFridges(3)["ces_budget_fridges"]["Lb"]
        Lcr = self.CESFridges(3)["ces_budget_fridges"]["Lc"]
        Kbr = self.CESFridges(3)["ces_budget_fridges"]["Kb"]
        Kcr = self.CESFridges(3)["ces_budget_fridges"]["Kc"]

        Lbb = self.CESFridges(4)["ces_budget_fridges"]["Lb"]
        Lcb = self.CESFridges(4)["ces_budget_fridges"]["Lc"]
        Kbb = self.CESFridges(4)["ces_budget_fridges"]["Kb"]
        Kcb = self.CESFridges(4)["ces_budget_fridges"]["Kc"]



        # NONENERGY SECTORS ---------------------------------------
        # Base Run Equilibrium-------------------------------------

        #Axes1
        self.ces_budget_0_x_base = np.linspace(0, Lb0, 100)
        self.ces_budget_0_y_base = self.CESBudget(self.ces_budget_0_x_base, 0)["Base"]

        X = np.linspace(0, Lb0, 100)
        Y = self.CES(X,0)["Base"]

        self.ces_0_x_base = []
        self.ces_0_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb0 or Y[i] < 0:
                continue

            self.ces_0_x_base.append(X[i])
            self.ces_0_y_base.append(Y[i])


        #Axes2
        self.ces_budget_1_x_base = np.linspace(0, Lb1, 100)
        self.ces_budget_1_y_base = self.CESBudget(self.ces_budget_1_x_base, 1)["Base"]

        X = np.linspace(0, Lb1, 111)
        Y = self.CES(X,1)["Base"]

        self.ces_1_x_base = []
        self.ces_1_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb1 or Y[i] < 0:
                continue

            self.ces_1_x_base.append(X[i])
            self.ces_1_y_base.append(Y[i])


        #Axes 3
        self.ces_budget_2_x_base = np.linspace(0, Lb2, 100)
        self.ces_budget_2_y_base = self.CESBudget(self.ces_budget_2_x_base, 2)["Base"]

        X = np.linspace(0, Lb2, 222)
        Y = self.CES(X,2)["Base"]

        self.ces_2_x_base = []
        self.ces_2_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kb2 or Y[i] < 0:
                continue

            self.ces_2_x_base.append(X[i])
            self.ces_2_y_base.append(Y[i])



        # NONENERGY SECTORS ---------------------------------------
        # CFRUN Equilibrium----------------------------------------

        #Axes1
        self.ces_budget_0_x_cfrun = np.linspace(0, Lc0, 100)
        self.ces_budget_0_y_cfrun = self.CESBudget(self.ces_budget_0_x_cfrun, 0)["CFRun"]

        X = np.linspace(0, Lc0, 100)
        Y = self.CES(X,0)["CFRun"]

        self.ces_0_x_cfrun = []
        self.ces_0_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc0 or Y[i] < 0:
                continue

            self.ces_0_x_cfrun.append(X[i])
            self.ces_0_y_cfrun.append(Y[i])


        #Axes2
        self.ces_budget_1_x_cfrun = np.linspace(0, Lc1, 100)
        self.ces_budget_1_y_cfrun = self.CESBudget(self.ces_budget_1_x_cfrun, 1)["CFRun"]

        X = np.linspace(0, Lc1, 111)
        Y = self.CES(X,1)["CFRun"]

        self.ces_1_x_cfrun = []
        self.ces_1_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc1 or Y[i] < 0:
                continue

            self.ces_1_x_cfrun.append(X[i])
            self.ces_1_y_cfrun.append(Y[i])


        #Axes3
        self.ces_budget_2_x_cfrun = np.linspace(0, Lc2, 100)
        self.ces_budget_2_y_cfrun = self.CESBudget(self.ces_budget_2_x_cfrun, 2)["CFRun"]

        X = np.linspace(0, Lc2, 222)
        Y = self.CES(X,2)["CFRun"]

        self.ces_2_x_cfrun = []
        self.ces_2_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kc2 or Y[i] < 0:
                continue

            self.ces_2_x_cfrun.append(X[i])
            self.ces_2_y_cfrun.append(Y[i])



        # ENERGY SECTORS ------------------------------------------
        # Base Run Equilibrium-------------------------------------

        #Axes1
        self.ces_budget_r_x_base = np.linspace(0, Lbr, 100)
        self.ces_budget_r_y_base = self.CESBudget(self.ces_budget_r_x_base, 3)["Base"]

        X = np.linspace(0, Lbr, 100)
        Y = self.CES(X,3)["Base"]

        self.ces_r_x_base = []
        self.ces_r_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kbr or Y[i] < 0:
                continue

            self.ces_r_x_base.append(X[i])
            self.ces_r_y_base.append(Y[i])

        #Axes2
        self.ces_budget_b_x_base = np.linspace(0, Lbb, 100)
        self.ces_budget_b_y_base = self.CESBudget(self.ces_budget_b_x_base, 4)["Base"]

        X = np.linspace(0, Lbb, 100)
        Y = self.CES(X,4)["Base"]

        self.ces_b_x_base = []
        self.ces_b_y_base = []

        for i in range(len(Y)):
            if Y[i] > Kbb or Y[i] < 0:
                continue

            self.ces_b_x_base.append(X[i])
            self.ces_b_y_base.append(Y[i])



        # ENERGY SECTORS ---------------------------------------
        # CFRUN Equilibrium----------------------------------------

        #Axes1
        self.ces_budget_r_x_cfrun = np.linspace(0, Lcr, 100)
        self.ces_budget_r_y_cfrun = self.CESBudget(self.ces_budget_r_x_cfrun, 3)["CFRun"]

        X = np.linspace(0, Lcr, 100)
        Y = self.CES(X,3)["CFRun"]

        self.ces_r_x_cfrun = []
        self.ces_r_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kcr or Y[i] < 0:
                continue

            self.ces_r_x_cfrun.append(X[i])
            self.ces_r_y_cfrun.append(Y[i])


        #Axes2
        self.ces_budget_b_x_cfrun = np.linspace(0, Lbb, 100)
        self.ces_budget_b_y_cfrun = self.CESBudget(self.ces_budget_b_x_cfrun, 4)["CFRun"]

        X = np.linspace(0, Lbb, 100)
        Y = self.CES(X,4)["CFRun"]

        self.ces_b_x_cfrun = []
        self.ces_b_y_cfrun = []

        for i in range(len(Y)):
            if Y[i] > Kbb or Y[i] < 0:
                continue

            self.ces_b_x_cfrun.append(X[i])
            self.ces_b_y_cfrun.append(Y[i])

    def NonEnergyCESPlot(self):
        
        self.CesValues()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15, 4))


        #-------------------------Base Run Equilibrium-------------------------------------

        #AGR
        label_base_ces_0 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[0], 
                                                                                              self.L_base[0], self.K_base[0], 
                                                                                              1/self.r_base)

        ax1.plot( self.ces_budget_0_x_base, self.ces_budget_0_y_base,  "-k", linewidth = 3, label = label_base_ces_0)
        ax1.plot( self.ces_0_x_base, self.ces_0_y_base, "-k", linewidth = 3)

        ax1.plot([0, self.L_base[0]], [self.K_base[0], self.K_base[0]], "--ok")
        ax1.plot([self.L_base[0], self.L_base[0]], [0, self.K_base[0]], "--ok")

        #SERVICES
        label_base_ces_1 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[1], 
                                                                                              self.L_base[1], self.K_base[1], 
                                                                                              1/self.r_base)
        ax2.plot( self.ces_budget_1_x_base, self.ces_budget_1_y_base,  "-k", linewidth = 3, label = label_base_ces_1)
        ax2.plot( self.ces_1_x_base, self.ces_1_y_base, "-k", linewidth = 3)

        ax2.plot([0, self.L_base[1]], [self.K_base[1], self.K_base[1]], "--ok")
        ax2.plot([self.L_base[1], self.L_base[1]], [0, self.K_base[1]], "--ok")


        #INDUSTRY
        label_base_ces_2 =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[2], 
                                                                                              self.L_base[2], self.K_base[2], 
                                                                                              1/self.r_base)


        ax3.plot(self.ces_budget_2_x_base, self.ces_budget_2_y_base,  "-k", linewidth = 3, label = label_base_ces_2)
        ax3.plot(self.ces_2_x_base, self.ces_2_y_base, "-k", linewidth = 3)

        ax3.plot([0, self.L_base[2]], [self.K_base[2], self.K_base[2]], "--ok")
        ax3.plot([self.L_base[2], self.L_base[2]], [0, self.K_base[2]], "--ok")


        #-------------------------CFRun Equilibrium-------------------------------------
        
        #AGR
        label_cfrun_ces_0 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[0], 
                                                                                              self.L_cfrun[0], self.K_cfrun[0], 
                                                                                              1/self.r_cfrun)
        
        ax1.plot( self.ces_budget_0_x_cfrun, self.ces_budget_0_y_cfrun,  "--k", linewidth = 1, label = label_cfrun_ces_0)
        ax1.plot( self.ces_0_x_cfrun, self.ces_0_y_cfrun, "--k", linewidth = 1)

        ax1.plot([0, self.L_cfrun[0]], [self.K_cfrun[0], self.K_cfrun[0]], "--ok")
        ax1.plot([self.L_cfrun[0], self.L_cfrun[0]], [0, self.K_cfrun[0]], "--ok")


        #SERVICES
        label_cfrun_ces_1 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[1], 
                                                                                              self.L_cfrun[1], self.K_cfrun[1], 
                                                                                              1/self.r_cfrun)
        ax2.plot( self.ces_budget_1_x_cfrun, self.ces_budget_1_y_cfrun,  "--k", linewidth = 1, label = label_cfrun_ces_1)
        ax2.plot( self.ces_1_x_cfrun, self.ces_1_y_cfrun, "--k", linewidth = 1)

        ax2.plot([0, self.L_cfrun[1]], [self.K_cfrun[1], self.K_cfrun[1]], "--ok")
        ax2.plot([self.L_cfrun[1], self.L_cfrun[1]], [0, self.K_cfrun[1]], "--ok")


        #INDUSTRY
        label_cfrun_ces_2 =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[2], 
                                                                                              self.L_cfrun[2], self.K_cfrun[2], 
                                                                                              1/self.r_cfrun)
        ax3.plot( self.ces_budget_2_x_cfrun, self.ces_budget_2_y_cfrun,  "--k", linewidth = 1, label =label_cfrun_ces_2 )
        ax3.plot( self.ces_2_x_cfrun, self.ces_2_y_cfrun, "--k", linewidth = 1)

        ax3.plot([0, self.L_cfrun[2]], [self.K_cfrun[2], self.K_cfrun[2]], "--ok")
        ax3.plot([self.L_cfrun[2], self.L_cfrun[2]], [0, self.K_cfrun[2]], "--ok")






        ax1.set_title("{} Sektörü CES Üretim Fonksiyonu".format(self.sectors[0]),fontsize = 13)
        ax1.set_xlabel("Emek",fontweight = "bold", fontsize = 11)
        ax1.set_ylabel("Sermaye",fontweight = "bold", fontsize = 11)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.legend(fontsize = 7, edgecolor = "black", ncol = 2)


        #SERVICES
        ax2.set_xlim(xmin = 0)
        ax2.set_ylim(ymin = 0)
        ax2.set_title("{} Sektörü CES Üretim Fonksiyonu".format(self.sectors[1]), fontsize = 13)
        ax2.set_xlabel("Emek",fontweight = "bold", fontsize = 11)
        ax2.set_ylabel("Sermaye",fontweight = "bold", fontsize = 11)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.legend(fontsize = 7, edgecolor = "black", ncol = 2)



        #INDUSTRY
        ax3.set_xlim(xmin = 0)
        ax3.set_ylim(ymin = 0)
        ax3.set_title("{} Sektörü CES Üretim Fonksiyonu".format(self.sectors[2]), fontsize = 13)
        ax3.set_xlabel("Emek",fontweight = "bold", fontsize = 11)
        ax3.set_ylabel("Sermaye",fontweight = "bold", fontsize = 11)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.legend(fontsize = 7, edgecolor = "black", ncol = 2)

        plt.tight_layout()
        plt.show()

    def EnergyCESPlot(self):

        self.CesValues()

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))



        # ENERGY SECTORS ------------------------------------------
        # RAFİNERİLER----------------------------------------------

        ###### Base Run Equilibrium--------------------------------
        label_base_ces_r =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[3], 
                                                                                              self.L_base[3], self.K_base[3], 
                                                                                              1/self.r_base)

        ax1.plot( self.ces_budget_r_x_base, self.ces_budget_r_y_base,  "-k", linewidth = 3, label = label_base_ces_r)
        ax1.plot( self.ces_r_x_base, self.ces_r_y_base, "-k", linewidth = 3)

        ax1.plot([0, self.L_base[3]], [self.K_base[3], self.K_base[3]], "--ok")
        ax1.plot([self.L_base[3], self.L_base[3]], [0, self.K_base[3]], "--ok")


        ###### CF Run Equilibrium--------------------------------
        label_cfrun_ces_r =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[3], 
                                                                                              self.L_cfrun[3], self.K_cfrun[3], 
                                                                                              1/self.r_cfrun)

        ax1.plot( self.ces_budget_r_x_cfrun, self.ces_budget_r_y_cfrun,  "--k", linewidth = 1, label = label_cfrun_ces_r)
        ax1.plot( self.ces_r_x_cfrun, self.ces_r_y_cfrun, "--k", linewidth = 1)

        ax1.plot([0, self.L_cfrun[3]], [self.K_cfrun[3], self.K_cfrun[3]], "--ok")
        ax1.plot([self.L_cfrun[3], self.L_cfrun[3]], [0, self.K_cfrun[3]], "--ok")

        ###### Axes----------------------------------------------
        ax1.set_xlim(xmin = 0)
        ax1.set_ylim(ymin = 0)

        ax1.set_title("{} Sektörü CES Üretim Fonksiyonu".format(self.sectors[3]),fontsize = 13)
        ax1.set_xlabel("Emek",fontweight = "bold", fontsize = 11)
        ax1.set_ylabel("Sermaye",fontweight = "bold", fontsize = 11)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.legend(fontsize = 7, edgecolor = "black", ncol = 2)



        # ENERGY SECTORS ------------------------------------------
        # BOTAŞ----------------------------------------------------

        ###### Base Run Equilibrium--------------------------------
        label_base_ces_b =  "Base Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_base[4], 
                                                                                              self.L_base[4], self.K_base[4], 
                                                                                              1/self.r_base)

        ax2.plot( self.ces_budget_b_x_base, self.ces_budget_b_y_base,  "-k", linewidth = 3, label = label_base_ces_b)
        ax2.plot( self.ces_b_x_base, self.ces_b_y_base, "-k", linewidth = 3)

        ax2.plot([0, self.L_base[4]], [self.K_base[4], self.K_base[4]], "--ok")
        ax2.plot([self.L_base[4], self.L_base[4]], [0, self.K_base[4]], "--ok")


        ###### CF Run Equilibrium--------------------------------
        label_cfrun_ces_b =  "cfrun Year\nX: {:.2f}\nL: {:.2f}\nK: {:.2f}\nw/r: {:.2f}".format(self.X_cfrun[4], 
                                                                                              self.L_cfrun[4], self.K_cfrun[4], 
                                                                                              1/self.r_cfrun)

        ax2.plot( self.ces_budget_b_x_cfrun, self.ces_budget_b_y_cfrun,  "--k", linewidth = 1, label = label_cfrun_ces_b)
        ax2.plot( self.ces_b_x_cfrun, self.ces_b_y_cfrun, "--k", linewidth = 1)

        ax2.plot([0, self.L_cfrun[4]], [self.K_cfrun[4], self.K_cfrun[4]], "--ok")
        ax2.plot([self.L_cfrun[4], self.L_cfrun[4]], [0, self.K_cfrun[4]], "--ok")

        ###### Axes----------------------------------------------
        ax2.set_xlim(xmin = 0)
        ax2.set_ylim(ymin = 0)

        ax2.set_title("{} Sektörü CES Üretim Fonksiyonu".format(self.sectors[4]),fontsize = 13)
        ax2.set_xlabel("Emek",fontweight = "bold", fontsize = 11)
        ax2.set_ylabel("Sermaye",fontweight = "bold", fontsize = 11)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.legend(fontsize = 7, edgecolor = "black", ncol = 2)


        plt.tight_layout()
        plt.show()

    def EV_CV_Calculation(self):

        EV = (self.U_cfrun - self.U_base) / self.U_base * self.Yd_base
        CV = (self.U_cfrun - self.U_base) / self.U_cfrun * self.Yd_cfrun
        

        print("U1: {}\nU2 {}".format(round(self.U_base,2), round(self.U_cfrun,2)))
        print("EV: {}\nCV {}".format(round(EV,2), round(CV,2)))

    def MacroVariables(self):

        df = pd.DataFrame(index = self.index)
        df["Base Year"] = self.base_values
        df["CF Year"] = self.cfrun_values
        df["%Change"] = (np.array(self.cfrun_values) - np.array(self.base_values))/ np.array(self.base_values) * 100
        # print(df)
        # df.to_excel("MacroVariables.xlsx")

        return df
 
    def flow_diagram(self, sector_code):

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x]
    
        # Sektör kodu = 1: Tarım, 2: Ticaret ve hizmet, 3: Sanayi
        
        sectors = {1: "Tarım", 2: "Ticaret Hizmet", 3:"Sanayi"}
        
        values_base = []
        values_cfrun = []
        
        position = [(1,17), (3,17), (2,15), (5,13), 
                (2,10), (5,8), (2,8), (5,7), (2,6), 
                (1,4),
                (5,4)]
        
        variables1 = ["L", "K", "X", "I1", "I2", "I3", "Ir", "Ib",
                    "Z", "E", "D", "M", "Q", "C", "TPAO", "G", "INV",
                    "I_1", "I_2", "I_3", "I_r", "I_b"]
        
        variables = []
        for i in variables1:
            if "_" in i:
                val = i.replace("_", str(sector_code))
                variables.append(val)
            else:
                val = i + str(sector_code)
                variables.append(val)
                
        nodes1 = ["L", "K", "X", "I_cost",
                "Z", "E", "D", "M", "Q", "final",
                "I_income"]
    
        nodes = []
        for i in nodes1:
            if "_" in i:
                val = i.replace("_", str(sector_code))
                nodes.append(val)
            else:
                val = i + str(sector_code)
                nodes.append(val)
                

        Icost      = variables[3:8]
        Iincome    = variables[17:]
        final      = variables[13:17]
        
        
        for i in nodes:
            if i in CGE(SAM).init_values_str:
                values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

            else:
                values_base.append("")
                values_cfrun.append("")
                
        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val3_base = ""
        val3_cfrun = ""
        for y in final:
            val3_base = val3_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"  
            val3_cfrun = val3_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
                
        
        edges1 = [("L", "X"), ("K", "X"),
        ("X", "Z"), ("I_cost", "Z"), ("Z", "E"),("Z", "D"), ("D", "Q"), ("M", "Q"),
        ("Q","final"),
        ("Q", "I_income")]
        
        edges = []
        
        for i in edges1:
            k = []
            
            for j in i:
                
                if "_" in j:
                    val = j.replace("_", str(sector_code))
                    k.append(val)
                else:
                    val = j + str(sector_code)
                    k.append(val)
            edges.append(tuple(k))

        G1 = nx.Graph()  
        G2 = nx.Graph()
            
        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])
        
        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')
        
        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')
        
        G1.add_edges_from(edges)
        G2.add_edges_from(edges)
        
            
        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);
        
        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);
        

        ax1.text(5,11, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(5,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(.5,1.7, val3_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        
        ax2.text(5,11, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(5,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(.5,1.7, val3_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
            
        
        ax1.set_title("{} Sektörü Baz Yıl Verileri".format(sectors[sector_code]), fontweight = "bold", 
                    color = "grey")
        ax2.set_title("{} Sektörü Karşı Olgusal Denge Verileri".format(sectors[sector_code]), fontweight = "bold",
                    color = "grey")
        
        ax1.set_xlim(xmin = 0, xmax = 6.5)
        ax1.set_ylim(ymin = 0, ymax = 18)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        
        ax2.set_xlim(xmin = 0, xmax = 6.5)
        ax2.set_ylim(ymin = 0, ymax = 18)
        
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
            
        plt.tight_layout() 
        plt.show()

    def flow_raf(self):

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x]
       
        values_base = []
        values_cfrun = []

        position = [(1,19), (3,19), (2,17), 
                    (5,19), (7,19), (6,17),
                    (4,15), (5,12),
                    (2,12), (1,10), (3,10), (3,8), (1,6),
                    (1,4), (3,3)             
                    ]

        variables = ["Lr", "Kr", "Xr",
                    "MCOr", "DCOr", "COr",
                    "XCOr", "I1r", "I2r", "I3r", "Irr", "Ibr",
                    "Zr", "Dr", "Er", "Mr", "Qr",
                    "Cr", "Ir1", "Ir2", "Ir3", "Irr"]

        nodes =  ["Lr", "Kr", "Xr",
                "MCOr", "DCOr", "COr",
                "XCOr", "I_cost",
                "Zr", "Dr", "Er", "Mr", "Qr",
                "Cr", "I_income"]

        Icost      = variables[7:12]
        Iincome    = variables[18:]



        for i in nodes:
                if i in CGE(SAM).init_values_str:
                    values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                    values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

                else:
                    values_base.append("")
                    values_cfrun.append("")

        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        edges = [
                ("Lr", "Xr"), ("Kr", "Xr"), ("MCOr", "COr"), ("DCOr", "COr"), ("Xr", "XCOr"), ("COr", "XCOr"),
                ("XCOr", "Zr"), ("I_cost", "Zr"), ("Zr", "Dr"), ("Zr", "Er"), ("Dr", "Qr"), ("Mr", "Qr"),("Qr", "Cr"), ("Qr", "I_income")

                ]   

        G1 = nx.Graph()  
        G2 = nx.Graph()

        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])

        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')

        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')

        G1.add_edges_from(edges)
        G2.add_edges_from(edges)


        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);

        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);


        ax1.text(5,9, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(3,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")

        ax2.text(5,9, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(3,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")



        ax1.set_title("Rafineriler Baz Yıl Verileri", fontweight = "bold", 
                    color = "grey")
        ax2.set_title("Rafineriler Karşı Olgusal Denge Verileri", fontweight = "bold",
                    color = "grey")

        ax1.set_xlim(xmin = 0, xmax = 8)
        ax1.set_ylim(ymin = 0, ymax = 20)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)

        ax2.set_xlim(xmin = 0, xmax = 8)
        ax2.set_ylim(ymin = 0, ymax = 20)

        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        plt.tight_layout()  
        plt.show()

    def flow_botas(self):  

        BaserunResult = [round(float(x), 3) for x in self.result1.x]
        CfrunResult   = [round(float(x), 3) for x in self.result2.x] 

        values_base = []
        values_cfrun = []

        position = [(1,19), (3,19), (2,17), 
                    (5,19), (7,19), (6,17),
                    (4,15), (5,12),
                    (2,12), (1,8), (3,8),
                    (1,4), (3,3)             
                    ]

        variables = ["Lb", "Kb", "Xb",
                    "MNGb", "DNGb", "NGb",
                    "XNGb", "I1b", "I2b", "I3b", "Ibb",
                    "Zb", "Db", "Eb", 
                    "Cb", "Ib1", "Ib2", "Ib3",  "Ibr"]

        nodes =  ["Lb", "Kb", "Xb",
                "MNGb", "DNGb", "NGb",
                "XNGb", "I_cost",
                "Zb", "Db", "Eb",
                "Cb", "I_income"]

        Icost      = variables[7:11]
        Iincome    = variables[15:]



        for i in nodes:
                if i in CGE(SAM).init_values_str:
                    values_base.append("{}: {}".format(i, BaserunResult[(CGE(SAM).init_values_str.index(i))]))
                    values_cfrun.append("{}: {}".format(i, CfrunResult[(CGE(SAM).init_values_str.index(i))]))

                else:
                    values_base.append("")
                    values_cfrun.append("")

        val1_base = ""
        val1_cfrun = ""
        for y in Icost:
            val1_base = val1_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"
            val1_cfrun = val1_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        val2_base = ""
        val2_cfrun = ""
        for y in Iincome:
            val2_base = val2_base + y + ": " + str(BaserunResult[(CGE(SAM).init_values_str.index(y))])+ "\n"
            val2_cfrun = val2_cfrun + y + ": " + str(CfrunResult[(CGE(SAM).init_values_str.index(y))]) + "\n"


        edges = [
                ("Lb", "Xb"), ("Kb", "Xb"), ("MNGb", "NGb"), ("DNGb", "NGb"), ("Xb", "XNGb"), ("NGb", "XNGb"),
                ("XNGb", "Zb"), ("I_cost", "Zb"), ("Zb", "Db"), ("Zb", "Eb"), ("Db", "Cb"), ("Db", "I_income")

                ]   

        G1 = nx.Graph()  
        G2 = nx.Graph()

        for k in range(len(nodes)):
            G1.add_node(nodes[k], pos = position[k], val = values_base[k])
            G2.add_node(nodes[k], pos = position[k], val = values_cfrun[k])

        pos1 = nx.get_node_attributes(G1,'pos')
        pos2 = nx.get_node_attributes(G2,'pos')

        labels1 = nx.get_node_attributes(G1,'val')
        labels2 = nx.get_node_attributes(G2,'val')

        G1.add_edges_from(edges)
        G2.add_edges_from(edges)


        fig, (ax1, ax2) = plt.subplots(1,2,figsize = (10,6))

        nx.draw_networkx_nodes(G1, pos1, node_size = 1500, node_color = "none", ax = ax1)
        nx.draw_networkx_edges(G1, pos1, edgelist = G1.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax1);
        nx.draw_networkx_labels(G1, pos1,labels = labels1, font_weight = "bold", ax = ax1);

        nx.draw_networkx_nodes(G2, pos2, node_size = 1500, node_color = "none", ax = ax2)
        nx.draw_networkx_edges(G2, pos2, edgelist = G2.edges(), edge_color = "black", arrows = True,
                        arrowstyle = "wedge", alpha = 0.2, arrowsize = 15, ax = ax2);
        nx.draw_networkx_labels(G2, pos2,labels = labels2, font_weight = "bold", ax = ax2);


        ax1.text(5,9, val1_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax1.text(3,1, val2_base, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")

        ax2.text(5,9, val1_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")
        ax2.text(3,1, val2_cfrun, fontsize = 10, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'),fontweight = "bold")

        ax1.set_title("BOTAŞ Baz Yıl Verileri", fontweight = "bold", 
                    color = "grey")
        ax2.set_title("BOTAŞ Karşı Olgusal Denge Verileri", fontweight = "bold",
                    color = "grey")

        ax1.set_xlim(xmin = 0, xmax = 8)
        ax1.set_ylim(ymin = 0, ymax = 20)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)

        ax2.set_xlim(xmin = 0, xmax = 8)
        ax2.set_ylim(ymin = 0, ymax = 20)

        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        plt.tight_layout()     
        plt.show()  
    
    def FinalConsumption(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (9,6))

        

        HouseholdConsumption = list((np.array(self.C_cfrun) - np.array(self.C_base)) / np.array(self.C_base) *100)
        HouseholdConsumption.insert(0, (self.HouseholdCfrunIncome - self.HouseholdBaseIncome) / self.HouseholdBaseIncome * 100 )
       
       
        ind = np.arange(0, len(HouseholdConsumption))

        ax1.bar(ind, HouseholdConsumption, width = 0.5, edgecolor = "k", 
                color = "gainsboro", label = "Hanehalkı Gelir ve\nTüketim Değişimleri (%)" )
        
        ax1.set_xticks(ind)
        ax1.set_xticklabels([ "Faktör Geliri", "Tarım", "Hizmet", "Sanayi", "Raf", "BOTAŞ"], fontsize = 9, fontweight= "bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(ncol = 2, loc = 2, edgecolor = "k", fontsize = 8)
        ax1.axhline(y=0, lw = 1, ls = "--", color = "k")
        ax1.set_title("Panel A", fontweight = "bold", fontsize = 10)


        GovernmentConsumption = list((np.array(self.G_cfrun) - np.array(self.G_base)) / np.array(self.G_base) *100)
        GovernmentConsumption.insert(0, (self.GovCfrunIncome - self.GovBaseIncome) / self.GovBaseIncome * 100 )

        ind = np.arange(0, len(GovernmentConsumption))
        ax2.bar(ind, GovernmentConsumption, width = 0.4, edgecolor = "k", 
                color = "gainsboro", label = "Kamu Gelir ve\nTüketimDeğişimi (%)" )
        
        ax2.set_xticks(ind)
        ax2.set_xticklabels(["Vergi Geliri","Tarım", "Hizmet", "Sanayi"], fontsize = 9, fontweight= "bold")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(ncol = 2, loc = 2, edgecolor = "k", fontsize = 8)
        # ax1.set_ylim(ymax = int(np.max(vergi)) * 1.2)
        ax2.set_title("Panel B", fontweight = "bold", fontsize = 10)
        ax2.axhline(y=0, lw = 1, ls = "--", color = "k")


        InvestmentConsumption = list((np.array(self.INV_cfrun) - np.array(self.INV_base)) / np.array(self.INV_base) *100)
        InvestmentConsumption.insert(0, (self.SavingsCfrun - self.SavingsBase) / self.SavingsBase * 100 )




        ind = np.arange(0, len(InvestmentConsumption))
        ax3.bar(ind, InvestmentConsumption, width = 0.4, edgecolor = "k", 
                color = "gainsboro", label = "Tasarruf - Yatırım\nHarcamaları Değişimi (%)" )
        
        ax3.set_xticks(ind)
        ax3.set_xticklabels(["Tasarruflar", "Tarım", "Hizmet", "Sanayi"], fontsize = 9, fontweight= "bold")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.legend(ncol = 2, loc = 4, edgecolor = "k", fontsize = 8)
        ax3.axhline(y=0, lw = 1, ls = "--", color = "k")
        ax3.set_title("Panel C", fontweight = "bold", fontsize = 10)


        TPAOConsumption = list((np.array(self.TPAO_cfrun) - np.array(self.TPAO_base)) / np.array(self.TPAO_base) *100)
        TPAOConsumption.insert(0, (self.EIncomeCfrun - self.EIncomeBase) / self.EIncomeBase * 100 )


        ind = np.arange(0, len(TPAOConsumption))
        ax4.bar(ind, TPAOConsumption, width = 0.4, edgecolor = "k", 
                color = "gainsboro", label = "Enerji Gelir ve\nHarcamalarıDeğişimi (%)" )
        
        ax4.set_xticks(ind)
        ax4.set_xticklabels(["Enerji Gelirleri", "Tarım", "Hizmet", "Sanayi"], fontsize = 9, fontweight= "bold")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.legend(ncol = 2, loc = 2, edgecolor = "k", fontsize = 8)
        ax4.axhline(y=0, lw = 1, ls = "--", color = "k")
        ax4.set_title("Panel D", fontweight = "bold", fontsize = 10)
        ax4.set_ylim(ymax = 2500)
        plt.tight_layout()
        plt.show()

    def PriceChanges(self):

        fig, ax1 = plt.subplots(figsize = (9,3))

        pricechanges = (np.array(self.PriceCfrun) - np.array(self.PriceBase)) / np.array(self.PriceBase) * 100
        ind = np.arange(0, len(pricechanges))
        labels = ["px1", "px2", "px3",  "pxr", "pxb", 
            "pz1", "pz2", "pz3", "pzr", "pzb", "pe1", "pe2", "pe3", 
            "per", "peb", "pd1", "pd2", "pd3", "pdr", "pdb", 
            "pq1", "pq2", "pq3",  "pqr", "pm1", "pm2", "pm3", 
            "pmr", "pmco",  "pco", "pxco", "pmng", "png", "pxng", 
             "r"]



        ax1.bar(ind, pricechanges, width = 0.5, edgecolor = "k", 
                color = "gainsboro", label = "Karşı Olgusal Denge\nFiyat Değişimleri (%)" )
        
        ax1.set_xticks(ind)
        ax1.set_xticklabels(labels, fontsize = 9, fontweight= "bold", rotation = 90)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(ncol = 1, loc = 3, edgecolor = "k", fontsize = 8)
        ax1.axhline(y=0, lw = 1, ls = "--", color = "k")
        # ax1.set_title("Panel A", fontweight = "bold", fontsize = 10)


        plt.tight_layout()
        plt.show()

    def SAMPlot(self):

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')


        BaseSHM = SAM
        CFSHM1 = self.CFSHM
        CFSHM2 = self.CFRunQuantitiySAM


        ChangeSHM1 = pd.DataFrame(index = BaseSHM.index, columns = BaseSHM.columns)
        ChangeSHM2 = pd.DataFrame(index = BaseSHM.index, columns = BaseSHM.columns)

        for row in ChangeSHM1.index:
            for col in ChangeSHM1.columns:
                baseValue = BaseSHM.loc[row, col]
                cfValue = CFSHM1.loc[row,col]

                if isinstance(baseValue, float) and isinstance(cfValue, float) and baseValue != 0:
                    difference = round((cfValue - baseValue )/ baseValue *100, 2)
                    ChangeSHM1.loc[row,col] = difference

                else:
                    if baseValue == 0 and cfValue == 0:
                        ChangeSHM1.loc[row,col] = 0

                    elif baseValue == 0 and cfValue > 1:
                        ChangeSHM1.loc[row,col] = 100
                    
                    else: 
                        ChangeSHM1.loc[row,col] = 0

        for row in ChangeSHM2.index:
            for col in ChangeSHM2.columns:
                baseValue = BaseSHM.loc[row, col]
                cfValue = CFSHM2.loc[row,col]

                if isinstance(baseValue, float) and isinstance(cfValue, float) and baseValue != 0:
                    difference = round((cfValue - baseValue )/ baseValue *100, 2)
                    ChangeSHM2.loc[row,col] = difference

                else:
                    if baseValue == 0 and cfValue == 0:
                        ChangeSHM2.loc[row,col] = 0

                    elif baseValue == 0 and cfValue > 1:
                        ChangeSHM2.loc[row,col] = 100
                    
                    else: 
                        ChangeSHM2.loc[row,col] = 0

        xpos1, ypos1 = np.meshgrid(np.arange(ChangeSHM1.shape[1]), np.arange(ChangeSHM1.shape[0]))
        xpos1 = xpos1.flatten()
        ypos1 = ypos1.flatten()
        zpos1 = np.zeros_like(xpos1)

        dx1= dy1 = .8
        dz1 = ChangeSHM1.values.flatten()
        dz1_ = np.copy(dz1)
        dz1_[dz1_ < 0] = 0
        
        colors = np.where(dz1_ > 0, 'blue', 'white')
        ax1.bar3d(xpos1, ypos1, zpos1, dx1, dy1, dz1_, color=colors)
        ax1.set_xticks(np.arange(ChangeSHM1.shape[1]))
        ax1.set_xticklabels([])
        ax1.set_yticks(np.arange(ChangeSHM1.shape[1]))
        ax1.set_yticklabels([])
        ax1.set_xlabel('Sütun')
        ax1.set_ylabel('Satır')
        ax1.set_zlabel('% Değişim')
        ax1.set_xlim(xmin = 0)
        ax1.set_ylim(ymin = 0)
        ax1.set_zlim(zmin = 0)
        ax1.set_title("Değer Değişimi [Artış Olanlar]", fontweight = "bold")
        ax1.grid(False)


        dz2 = np.copy(dz1)
        dz2[dz2 > 0] = 0
        dz2 = dz2*-1


        colors = np.where(dz2 > 0, 'red', 'white')
        ax2.bar3d(xpos1, ypos1, zpos1, dx1, dy1, dz2, color=colors)
        ax2.set_xticks(np.arange(ChangeSHM1.shape[1]))
        ax2.set_xticklabels([])
        ax2.set_yticks(np.arange(ChangeSHM1.shape[1]))
        ax2.set_yticklabels([])

        ax2.set_xlabel('Sütun')
        ax2.set_ylabel('Satır', rotation = -45)
        ax2.set_zlabel('% Değişim')
        ax2.set_xlim(xmin = 0)
        ax2.set_ylim(ymin = 0)
        ax2.set_zlim(zmin = 0)
        ax2.set_title("Değer Değişimi [Azalış Olanlar]", fontweight = "bold")
        ax2.grid(False)




        #########################################
        #########################################

        xpos2, ypos2 = np.meshgrid(np.arange(ChangeSHM2.shape[1]), np.arange(ChangeSHM2.shape[0]))
        xpos2 = xpos2.flatten()
        ypos2 = ypos2.flatten()
        zpos2 = np.zeros_like(xpos2)

        dx2= dy2 = .8
        dz2 = ChangeSHM2.values.flatten()
        dz2_ = np.copy(dz2)
        dz2_[dz2_ < 0] = 0
        
        colors = np.where(dz2_ > 0, 'blue', 'white')
        ax3.bar3d(xpos2, ypos2, zpos2, dx2, dy2, dz2_, color=colors)
        ax3.set_xticks(np.arange(ChangeSHM2.shape[1]))
        ax3.set_xticklabels([])
        ax3.set_yticks(np.arange(ChangeSHM2.shape[1]))
        ax3.set_yticklabels([])
        ax3.set_xlabel('Sütun')
        ax3.set_ylabel('Satır')
        ax3.set_zlabel('% Değişim')
        ax3.set_xlim(xmin = 0)
        ax3.set_ylim(ymin = 0)
        ax3.set_zlim(zmin = 0)
        ax3.set_title("Miktar Değişimi [Artış Olanlar]", fontweight = "bold")
        ax3.grid(False)


        dz2 = np.copy(dz2)
        dz2[dz2_ > 0] = 0
        dz2_ = dz2*-1


        colors = np.where(dz2_ > 0, 'red', 'white')
        ax4.bar3d(xpos2, ypos2, zpos2, dx2, dy2, dz2_, color=colors)
        ax4.set_xticks(np.arange(ChangeSHM2.shape[1]))
        ax4.set_xticklabels([])
        ax4.set_yticks(np.arange(ChangeSHM2.shape[1]))
        ax4.set_yticklabels([])
        ax4.set_xlabel('Sütun')
        ax4.set_ylabel('Satır')
        ax4.set_zlabel('% Değişim')
        ax4.set_xlim(xmin = 0)
        ax4.set_ylim(ymin = 0)
        ax4.set_zlim(zmin = 0)
        ax4.set_title("Miktar Değişimi [Azalış Olanlar]", fontweight = "bold")
        ax4.grid(False)

        
        
          
        
        
     
        plt.tight_layout()

        plt.show()


result = CGEResults(2)     # 0: Tarım, 1: Ticaret ve Hizmet, 2: Sanayi
# result.ArmingtonCETPlot()

# result.flow_diagram(2)     # 1: Tarım, 2: Ticaret ve Hizmet, 3: Sanayi
# result.flow_raf()
# result.flow_botas()
# result.NonEnergyCESPlot()
print(result.MacroVariables())
# result.EV_CV_Calculation()
# result.FinalConsumption()
# result.PriceChanges()
# result.EnergyCESPlot()
