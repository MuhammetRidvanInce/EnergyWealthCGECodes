from scipy.optimize import minimize
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class CGE():
    
    def __init__(self, SAM):  
        self.SAM = SAM
        shm = self.SAM.loc

        # DIŞŞSAl DEĞİŞKENLER

        ## Üretim Faktörleri
        self.DCOBar = shm["dco", "raf"] 
        self.DNGBar = shm["dng", "bts"]
        self.Lbar   = shm["hh", "lab"]
        self.Kbar   = shm["hh", "cap"]
        self.E_Energy = shm["eexp", "exp"]

        ## Yabancı Tasarruf # SF Yabancı Dışsal Değişken Değil

        ## Dünya Fiyatları
        self.Pwe1, self.Pwe2, self.Pwe3, self.Pwer, self.Pweb, \
        self.Pwm1, self.Pwm2, self.Pwm3, self.Pwmr, \
        self.Pwmco, self.Pwmng, self.epsilon = np.ones(12)

        ## Dönüşüm esneklikleri
        self.psi1 = 1.12
        self.psi2 = 0.5
        self.psi3 = 0.601
        self.psir = 10
        self.psib = 10 
        
        ## Armington İkame Esneklikleri
        self.sigma1   = 1.81
        self.sigma2   = 0.8
        self.sigma3   = 0.4
        self.sigmar   = 10 # Rafineriler D / M arasındaki ikame petrol ürünleri
        self.sigmaco  = 10 # İthal hampetrol / yurtiçi hampetrol arasındaki ikame
        self.sigmang  = 10 # İthal doğalgaz / yurtiçi doğalgaz arasındaki ikame



        ## CES İkame Esnekliği
        self.omega1 = 0.678
        self.omega2 = 0.4
        self.omega3 = 0.678
        self.omegar = 0.678 # Sanayi ile aynı olduğu düşünülmüştür.
        self.omegab = 0.678 # Sanayi ile aynı olduğu düşünülmüştür.

        self.omegaxco = 0.01 # Kompozit faktör / ham petrol arasındaki ikame
        self.omegaxng = 0.01 # Kompozit faktör / ham petrol arasındaki ikame




        # BAŞLANGIÇ DEĞERLERİ (BAZ TIL DEĞERLERİ)
        ## Fiyatlar
        px1, px2, px3, pxr, pxb, \
        pz1, pz2, pz3, pzr, pzb, \
        pe1, pe2, pe3, per, peb, \
        pd1, pd2, pd3, pdr, pdb, \
        pq1, pq2, pq3, pqr, \
        pm1, pm2, pm3, pmr, pmco, pdco, \
        pco, pxco, pmng, pdng, png, pxng, r, w = np.ones(38)

        ## Diğer Başlangıç (Baz Yıl) Değerleri
        Y = w*self.Lbar + r*self.Kbar
        OIL_INCOME = pdco*self.DCOBar + pdng*self.DNGBar + self.epsilon*self.E_Energy
        L1 = shm["lab", "agr"]
        L2 = shm["lab", "ser"]
        L3 = shm["lab", "ind"]
        Lr = shm["lab", "raf"]
        Lb = shm["lab", "bts"]
        K1 = shm["cap", "agr"]
        K2 = shm["cap", "ser"]
        K3 = shm["cap", "ind"]
        Kr = shm["cap", "raf"]
        Kb = shm["cap", "bts"]
        X1 = L1 + K1
        X2 = L2 + K2
        X3 = L3 + K3
        Xr = Lr + Kr
        Xb = Lb + Kb
        MCOr = shm["mco","raf"]
        DCOr = shm["dco", "raf"]
        COr = MCOr + DCOr
        XCOr = Xr + COr
        MNGb = shm["mng", "bts"]
        DNGb = shm["dng", "bts"]
        NGb = MNGb + DNGb
        XNGb = Xb + NGb

        I11 = shm["agr", "agr"]
        I21 = shm["ser", "agr"]
        I31 = shm["ind", "agr"]
        Ir1 = shm["raf", "agr"]
        Ib1 = shm["bts", "agr"]

        I12 = shm["agr", "ser"]
        I22 = shm["ser", "ser"]
        I32 = shm["ind", "ser"]
        Ir2 = shm["raf", "ser"]
        Ib2 = shm["bts", "ser"]

        I13 = shm["agr", "ind"]
        I23 = shm["ser", "ind"]
        I33 = shm["ind", "ind"]
        Ir3 = shm["raf", "ind"]
        Ib3 = shm["bts", "ind"]

        I1r = shm["agr", "raf"]
        I2r = shm["ser", "raf"]
        I3r = shm["ind", "raf"]
        Irr = shm["raf", "raf"]
        Ibr = shm["bts", "raf"]

        I1b = shm["agr", "bts"]
        I2b = shm["ser", "bts"]
        I3b = shm["ind", "bts"]
        Irb = shm["raf", "bts"]
        Ibb = shm["bts", "bts"]

        Z1 = X1 + I11 + I21 + I31 + Ir1 + Ib1
        Z2 = X2 + I12 + I22 + I32 + Ir2 + Ib2
        Z3 = X3 + I13 + I23 + I33 + Ir3 + Ib3
        Zr = XCOr + I1r + I2r + I3r + Irr + Ibr
        Zb = XNGb + I1b + I2b + I3b + Irb + Ibb

        E1 = shm["agr", "exp"]
        E2 = shm["ser", "exp"]
        E3 = shm["ind", "exp"]
        Er = shm["raf", "exp"]
        Eb = shm["bts", "exp"]
        
        M1 = shm["imp", "agr"]
        M2 = shm["imp", "ser"]
        M3 = shm["imp", "ind"]
        Mr = shm["imp", "raf"]

        Td = shm["dtax", "hh"]

        Tva1 = shm["gtax", "agr"]
        Tva2 = shm["gtax", "ser"]
        Tva3 = shm["gtax", "ind"]
        Tvar = shm["gtax", "raf"]
        Tvab = shm["gtax", "bts"]
        Tva = Tva1 + Tva2 + Tva3 + Tvar + Tvab

        Tz1 = shm["ptax", "agr"]
        Tz2 = shm["ptax", "ser"]
        Tz3 = shm["ptax", "ind"]
        Tzr = shm["ptax", "raf"]
        Tzb = shm["ptax", "bts"]
        Tz = Tz1 + Tz2 + Tz3 +  Tzr + Tzb


        T = Td + Tz + Tva

        D1 = Z1 + (Tva1 + Tz1) - E1
        D2 = Z2 + (Tva2 + Tz2) - E2
        D3 = Z3 + (Tva3 + Tz3) - E3
        Dr = Zr + (Tvar + Tzr) - Er
        Db = Zb + (Tvab + Tzb) - Eb

        C1 = shm["agr", "hh"]
        C2 = shm["ser", "hh"]
        C3 = shm["ind", "hh"]
        Cr = shm["raf", "hh"]
        Cb = shm["bts", "hh"]

        TPAO1 = shm["agr", "TPAO"]
        TPAO2 = shm["ser", "TPAO"]
        TPAO3 = shm["ind", "TPAO"]
      
        G1 = shm["agr", "gov"]
        G2 = shm["ser", "gov"]
        G3 = shm["ind", "gov"]
     
        INV1 = shm["agr", "inv"]
        INV2 = shm["ser", "inv"]
        INV3 = shm["ind", "inv"]
      
        Q1 = C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 + I1r + I1b
        Q2 = C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 + I2r + I2b
        Q3 = C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 + I3r + I3b
        Qr = Cr + Ir1 + Ir2 + Ir3 +  Irr + Irb
        Db = Cb + Ib1 + Ib2 + Ib3 +  Ibr + Ibb

        Sp = shm["sav", "hh"]
        Sg = shm["sav", "gov"]
        Sf = shm["sav", "exp"]  # Sf içsel değişken.....
        S = Sp + Sg + Sf
        Yd = Y - Sp - Td
        
        # PARAMETRELERİN KALİBRASYONU
      
        self.td = Td / Y 
        self.tva1 = Tva1 / Z1
        self.tva2 = Tva2 / Z2
        self.tva3 = Tva3 / Z3
        self.tvar = Tvar / Zr
        self.tvab = Tvab / Zb

        self.tz1 = Tz1 / Z1
        self.tz2 = Tz2 / Z2
        self.tz3 = Tz3 / Z3
        self.tzr = Tzr / Zr
        self.tzb = Tzb / Zb

    
        self.alpha1 = (self.omega1 - 1 ) / self.omega1
        self.alpha2 = (self.omega2 - 1 ) / self.omega2
        self.alpha3 = (self.omega3 - 1 ) / self.omega3
        self.alphar = (self.omegar - 1 ) / self.omegar
        self.alphab = (self.omegab - 1 ) / self.omegab

        self.delta1 = L1**(1-self.alpha1) / (L1**(1-self.alpha1) + K1**(1-self.alpha1))
        self.delta2 = L2**(1-self.alpha2) / (L2**(1-self.alpha2) + K2**(1-self.alpha2))
        self.delta3 = L3**(1-self.alpha3) / (L3**(1-self.alpha3) + K3**(1-self.alpha3))
        self.deltar = Lr**(1-self.alphar) / (Lr**(1-self.alphar) + Kr**(1-self.alphar))
        self.deltab = Lb**(1-self.alphab) / (Lb**(1-self.alphab) + Kb**(1-self.alphab))

        self.beta1 = K1**(1-self.alpha1) / (L1**(1-self.alpha1) + K1**(1-self.alpha1))
        self.beta2 = K2**(1-self.alpha2) / (L2**(1-self.alpha2) + K2**(1-self.alpha2))
        self.beta3 = K3**(1-self.alpha3) / (L3**(1-self.alpha3) + K3**(1-self.alpha3))
        self.betar = Kr**(1-self.alphar) / (Lr**(1-self.alphar) + Kr**(1-self.alphar))
        self.betab = Kb**(1-self.alphab) / (Lb**(1-self.alphab) + Kb**(1-self.alphab))

        self.A1 = X1 / (self.delta1*L1**self.alpha1 + self.beta1*K1**self.alpha1)**(1/self.alpha1)
        self.A2 = X2 / (self.delta2*L2**self.alpha2 + self.beta2*K2**self.alpha2)**(1/self.alpha2)
        self.A3 = X3 / (self.delta3*L3**self.alpha3 + self.beta3*K3**self.alpha3)**(1/self.alpha3)
        self.Ar = Xr / (self.deltar*Lr**self.alphar + self.betar*Kr**self.alphar)**(1/self.alphar)
        self.Ab = Xb / (self.deltab*Lb**self.alphab + self.betab*Kb**self.alphab)**(1/self.alphab)

        self.alphaxco = (self.omegaxco - 1) / self.omegaxco
        self.xr   = Xr**(1-self.alphaxco) / (Xr**(1-self.alphaxco) + COr**(1-self.alphaxco))
        self.co   = COr**(1-self.alphaxco) / (Xr**(1-self.alphaxco) + COr**(1-self.alphaxco))
        self.Axco = XCOr / (self.xr*Xr**self.alphaxco + self.co*COr**self.alphaxco)**(1/self.alphaxco)

        self.alphaxng = (self.omegaxng - 1) / self.omegaxng
        self.xb   = Xb**(1-self.alphaxng) / (Xb**(1-self.alphaxng) + NGb**(1-self.alphaxng))
        self.ng   = NGb**(1-self.alphaxng) / (Xb**(1-self.alphaxng) + NGb**(1-self.alphaxng))
        self.Axng = XNGb / (self.xb*Xb**self.alphaxng + self.ng*NGb**self.alphaxng)**(1/self.alphaxng)

        self.a11 = I11 / Z1
        self.a21 = I21 / Z1
        self.a31 = I31 / Z1
        self.ar1 = Ir1 / Z1
        self.ab1 = Ib1 / Z1

        self.a12 = I12 / Z2
        self.a22 = I22 / Z2
        self.a32 = I32 / Z2
        self.ar2 = Ir2 / Z2
        self.ab2 = Ib2 / Z2

        self.a13 = I13 / Z3
        self.a23 = I23 / Z3
        self.a33 = I33 / Z3
        self.ar3 = Ir3 / Z3
        self.ab3 = Ib3 / Z3
      
        self.a1r = I1r / Zr
        self.a2r = I2r / Zr
        self.a3r = I3r / Zr
        self.arr = Irr / Zr
        self.abr = Ibr / Zr

        self.a1b = I1b / Zb
        self.a2b = I2b / Zb
        self.a3b = I3b / Zb
        self.arb = Irb / Zb
        self.abb = Ibb / Zb

        self.x1 = X1 / Z1
        self.x2 = X2 / Z2
        self.x3 = X3 / Z3
       
        self.xcor = XCOr / Zr
        self.xngb = XNGb / Zb

        self.rho1 = (self.psi1 + 1) / self.psi1
        self.rho2 = (self.psi2 + 1) / self.psi2
        self.rho3 = (self.psi3 + 1) / self.psi3
        self.rhor = (self.psir + 1) / self.psir
        self.rhob = (self.psib + 1) / self.psib

        self.eta1 = (self.sigma1 - 1) / self.sigma1
        self.eta2 = (self.sigma2 - 1) / self.sigma2
        self.eta3 = (self.sigma3 - 1) / self.sigma3
        self.etar = (self.sigmar - 1) / self.sigmar
        self.etaco = (self.sigmaco - 1) / self.sigmaco
        self.etang = (self.sigmang - 1) / self.sigmang

        self.e1 = E1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.e2 = E2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.e3 = E3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.er = Er**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.eb = Eb**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.dt1 = D1**(1-self.rho1) / (E1**(1-self.rho1) + D1**(1-self.rho1))
        self.dt2 = D2**(1-self.rho2) / (E2**(1-self.rho2) + D2**(1-self.rho2))
        self.dt3 = D3**(1-self.rho3) / (E3**(1-self.rho3) + D3**(1-self.rho3))
        self.dtr = Dr**(1-self.rhor) / (Er**(1-self.rhor) + Dr**(1-self.rhor))
        self.dtb = Db**(1-self.rhob) / (Eb**(1-self.rhob) + Db**(1-self.rhob))

        self.theta1 = Z1 / (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1)
        self.theta2 = Z2 / (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2)
        self.theta3 = Z3 / (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3)
        self.thetar = Zr / (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor)
        self.thetab = Zb / (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob)

        self.m1   = M1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.m2   = M2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.m3   = M3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.mr   = Mr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.mcor = MCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.mngb = MNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.m1   = M1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.m2   = M2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.m3   = M3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.mr   = Mr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.mcor = MCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.mngb = MNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.da1   = D1**(1-self.eta1) / (M1**(1-self.eta1) + D1**(1-self.eta1))
        self.da2   = D2**(1-self.eta2) / (M2**(1-self.eta2) + D2**(1-self.eta2))
        self.da3   = D3**(1-self.eta3) / (M3**(1-self.eta3) + D3**(1-self.eta3))
        self.dar   = Dr**(1-self.etar) / (Mr**(1-self.etar) + Dr**(1-self.etar))
        self.dcor  = DCOr**(1-self.etaco) / (MCOr**(1-self.etaco) + DCOr**(1-self.etaco))
        self.dngb  = DNGb**(1-self.etang) / (MNGb**(1-self.etang) + DNGb**(1-self.etang))

        self.lambda1 = Q1 / (self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1)
        self.lambda2 = Q2 / (self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2)
        self.lambda3 = Q3 / (self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3)
        self.lambdar = Qr / (self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar)
        self.lambdaco = COr / (self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco)
        self.lambdang = NGb / (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang)

        self.c1 = C1 / Yd
        self.c2 = C2 / Yd
        self.c3 = C3 / Yd
        self.cr = Cr / Yd
        self.cb = Cb / Yd

        self.mu1 = TPAO1 / OIL_INCOME 
        self.mu2 = TPAO2 / OIL_INCOME 
        self.mu3 = TPAO3 / OIL_INCOME 
       
        self.g1 = G1 / (T - Sg)
        self.g2 = G2 / (T - Sg)
        self.g3 = G3 / (T - Sg)
       
        self.inv1 = INV1 / S
        self.inv2 = INV2 / S
        self.inv3 = INV3 / S
        
        self.sp = Sp / Y
        self.sg = Sg / T
        
        self.init_values =  [
            X1, L1, K1, I11, I21, I31, Ir1, Ib1, Z1,
            E1, D1, Q1, M1, X2, L2, K2, I12, I22, I32, 
            Ir2, Ib2, Z2, E2, D2, Q2, M2, X3, L3, K3, 
            I13, I23, I33, Ir3, Ib3, Z3, E3, D3, Q3, 
            M3, Xr, Lr, Kr, COr, MCOr, DCOr, XCOr, I1r, I2r, 
            I3r,  Irr, Ibr, Zr, Er, Dr, Qr, Mr, Xb, Lb, Kb, 
            NGb, MNGb, DNGb, XNGb, I1b, I2b, I3b,  Irb, 
            Ibb, Zb, Eb, Db, C1, C2, C3,  Cr, Cb, Y,
            Yd, TPAO1, TPAO2, TPAO3,  OIL_INCOME, G1, G2, 
            G3,  T, Td, Tz, Tva, Tz1, Tz2, Tz3, 
            Tzr, Tzb, Tva1, Tva2, Tva3,  Tvar, 
            Tvab, INV1, INV2, INV3,  
             S, Sp, Sg, px1, px2, px3,  pxr, pxb, 
            pz1, pz2, pz3, pzr, pzb, pe1, pe2, pe3, 
             per, peb, pd1, pd2, pd3, pdr, pdb, 
            pq1, pq2, pq3,  pqr, pm1, pm2, pm3, 
            pmr, pmco, pdco, pco, pxco, pmng, pdng, png, pxng, 
            Sf, r 
        ]
        
        self.init_values_str = [
            "X1", "L1", "K1", "I11", "I21", "I31", "Ir1", "Ib1", "Z1",
            "E1", "D1", "Q1", "M1", "X2", "L2", "K2", "I12", "I22", "I32", 
            "Ir2", "Ib2", "Z2", "E2", "D2", "Q2", "M2", "X3", "L3", "K3", 
            "I13", "I23", "I33", "Ir3", "Ib3", "Z3", "E3", "D3", "Q3", 
            "M3", "Xr", "Lr", "Kr", "COr", "MCOr", "DCOr", "XCOr", "I1r", "I2r", 
            "I3r",  "Irr", "Ibr", "Zr", "Er", "Dr", "Qr", "Mr", "Xb", "Lb", "Kb", 
            "NGb", "MNGb", "DNGb", "XNGb", "I1b", "I2b", "I3b",  "Irb", 
            "Ibb", "Zb", "Eb", "Db", "C1", "C2", "C3",  "Cr", "Cb", "Y",
            "Yd", "TPAO1", "TPAO2", "TPAO3",  "OIL_INCOME", "G1", "G2", 
            "G3",  "T", "Td", "Tz", "Tva",  "Tz1", "Tz2", "Tz3", 
            "Tzr", "Tzb", "Tva1", "Tva2", "Tva3",  "Tvar", 
            "Tvab", "INV1", "INV2", "INV3",  
             "S", "Sp", "Sg", "px1", "px2", "px3",  "pxr", "pxb", 
            "pz1", "pz2", "pz3", "pzr", "pzb", "pe1", "pe2", "pe3", 
            "per", "peb", "pd1", "pd2", "pd3", "pdr", "pdb", 
            "pq1", "pq2", "pq3",  "pqr", "pm1", "pm2", "pm3", 
            "pmr", "pmco", "pdco", "pco", "pxco", "pmng", "pdng", "png", "pxng", 
            "Sf", "r"    
        ]
        
        self.parameters_str=[
                "td", "tva1", "tva2", "tva3", "tvar", "tvab", "tz1", "tz2", "tz3", "tzr", "tzb",
                "alpha1", "alpha2", "alpha3", "alphar","alphab",
                "delta1", "delta2", "delta3", "deltar", "deltab", 
                "beta1", "beta2", "beta3", "betar", "betab",
                "A1", "A2", "A3", "Ar", "Ab",
                "Axco", "alphaxco", "xr", "co",
                "Axng", "alphaxng", "xb", "ng", 
                "a11", "a21", "a31", "ar1", "ab1", "a12", "a22", "a32", "ar2", "ab2", "a13", "a23",
                "a33", "ar3", "ab3", "a1r", "a2r", "a3r", "arr", "abr", "a1b", "a2b", "a3b", "arb", "abb", "x1", "x2",
                "x3", "xcor", "xngb", "rho1", "rho2", "rho3", "rhor", "rhob", "eta1", "eta2", "eta3", "etar", "etaco",
                "etang", "e1", "e2", "e3", "er", "eb", "dt1", "dt2", "dt3", "dtr", "dtb", "theta1", "theta2", "theta3",
                "thetar", "thetab", "m1", "m2", "m3", "mr", "mcor", "mngb", "m1", "m2", "m3", "mr", "mcor", "mngb", "da1",
                "da2", "da3", "dar", "dcor", "dngb", "lambda1", "lambda2", "lambda3", "lambdar", "lambdaco", "lambdang",
                "c1", "c2", "c3", "cr", "cb", "mu1", "mu2", "mu3", "g1", "g2", "g3", "inv1", "inv2", "inv3", "sp", "sg"
        ]
        
    def model_parameters(self):
        
        self.parameters=[
                self.td, self.tva1, self.tva2, self.tva3, self.tvar, self.tvab, self.tz1, self.tz2, self.tz3, self.tzr, self.tzb,
                self.alpha1, self.alpha2, self.alpha3, self.alphar,self.alphab, self.delta1, self.delta2,
                self.delta3, self.deltar, self.deltab, self.beta1, self.beta2, self.beta3,self.betar,self.betab, self.A1, self.A2, self.A3, 
                self.Ar, self.Ab, self.Axco, self.alphaxco, self.xr, self.co, self.Axng, self.alphaxng, self.xb, self.ng, 
                self.a11, self.a21, self.a31, self.ar1, self.ab1, self.a12, self.a22, self.a32, self.ar2, self.ab2, self.a13, self.a23,
                self.a33, self.ar3, self.ab3, self.a1r, self.a2r, self.a3r, self.arr, self.abr, self.a1b, self.a2b, self.a3b, self.arb, 
                self.abb, self.x1, self.x2, self.x3, self.xcor, self.xngb, self.rho1, self.rho2, self.rho3, self.rhor, self.rhob, self.eta1, 
                self.eta2, self.eta3, self.etar, self.etaco, self.etang, self.e1, self.e2, self.e3, self.er, self.eb, self.dt1, self.dt2, self.dt3, 
                self.dtr, self.dtb, self.theta1, self.theta2, self.theta3, self.thetar, self.thetab, self.m1, self.m2, self.m3, self.mr, self.mcor, 
                self.mngb, self.m1, self.m2, self.m3, self.mr, self.mcor, self.mngb, self.da1, self.da2, self.da3, self.dar, self.dcor, self.dngb, 
                self.lambda1, self.lambda2, self.lambda3, self.lambdar, self.lambdaco, self.lambdang, self.c1, self.c2, self.c3, self.cr, self.cb, 
                self.mu1, self.mu2, self.mu3, self.g1, self.g2, self.g3, self.inv1, self.inv2, self.inv3, self.sp, self.sg
            ]

        return self.parameters
        
    def objValue(self, x):
        
        C1 = x[71]
        C2 = x[72]
        C3 = x[73]
        Cr = x[74]
        Cb = x[75]

        return -C1**self.c1 * C2**self.c2 * C3**self.c3 * Cr**self.cr * Cb**self.cb
        
    def constraints(self, x):
        X1  = x[0]
        L1  = x[1]
        K1  = x[2]
        I11 = x[3]
        I21 = x[4]
        I31 = x[5]
        Ir1 = x[6]
        Ib1 = x[7]
        Z1  = x[8]
        E1  = x[9]
        D1  = x[10]
        Q1  = x[11]
        M1  = x[12]
        X2  = x[13]
        L2  = x[14]
        K2  = x[15]
        I12 = x[16]
        I22 = x[17]
        I32 = x[18]
        Ir2 = x[19]
        Ib2 = x[20]
        Z2  = x[21]
        E2  = x[22]
        D2  = x[23]
        Q2  = x[24]
        M2  = x[25]
        X3  = x[26]
        L3  = x[27]
        K3  = x[28]
        I13 = x[29]
        I23 = x[30]
        I33 = x[31]
        Ir3 = x[32]
        Ib3 = x[33]
        Z3  = x[34]
        E3  = x[35]
        D3  = x[36]
        Q3  = x[37]
        M3  = x[38]
        Xr   = x[39]
        Lr   = x[40]
        Kr   = x[41]
        COr  = x[42]
        MCOr = x[43]
        DCOr = x[44]
        XCOr = x[45]
        I1r  = x[46]
        I2r  = x[47]
        I3r  = x[48]
        Irr  = x[49]
        Ibr  = x[50]
        Zr   = x[51]
        Er   = x[52]
        Dr   = x[53]
        Qr   = x[54]
        Mr   = x[55]
        Xb   = x[56]
        Lb   = x[57]
        Kb   = x[58]
        NGb  = x[59]
        MNGb = x[60]
        DNGb = x[61]
        XNGb = x[62]
        I1b = x[63]
        I2b = x[64]
        I3b = x[65]
        Irb = x[66]
        Ibb = x[67]
        Zb  = x[68]
        Eb  = x[69]
        Db  = x[70]
        C1 = x[71]
        C2 = x[72]
        C3 = x[73]
        Cr = x[74]
        Cb = x[75]
        Y = x[76]
        Yd  = x[77]
        TPAO1 = x[78]
        TPAO2 = x[79]
        TPAO3 = x[80]
        OIL_INCOME = x[81]
        G1   = x[82]
        G2   = x[83]
        G3   = x[84]
        T    = x[85]
        Td   = x[86]
        Tz   = x[87]
        Tva  = x[88]
        Tz1  = x[89]
        Tz2  = x[90]
        Tz3  = x[91]
        Tzr  = x[92]
        Tzb  = x[93]
        Tva1 = x[94]
        Tva2 = x[95]
        Tva3 = x[96]
        Tvar = x[97]
        Tvab = x[98]
        INV1 = x[99]
        INV2 = x[100]
        INV3 = x[101]
        S    = x[102]
        Sp   = x[103]
        Sg   = x[104]
        px1  = x[105]
        px2  = x[106]
        px3  = x[107]
        pxr  = x[108]
        pxb  = x[109]
        pz1  = x[110]
        pz2  = x[111]
        pz3  = x[112]
        pzr  = x[113]
        pzb  = x[114]
        pe1  = x[115]
        pe2  = x[116]
        pe3  = x[117]
        per  = x[118]
        peb  = x[119]
        pd1  = x[120]
        pd2  = x[121]
        pd3  = x[122]
        pdr  = x[123]
        pdb  = x[124]
        pq1  = x[125]
        pq2  = x[126]
        pq3  = x[127]
        pqr  = x[128]
        pm1  = x[129]
        pm2  = x[130]
        pm3  = x[131]
        pmr  = x[132]
        pmco = x[133]
        pdco = x[134]
        pco  = x[135]
        pxco = x[136]
        pmng = x[137]
        pdng = x[138]
        png  = x[139]
        pxng = x[140]
        Sf = x[141]
        r    = x[142]
        w    = 1
        
        return [
        X1 - self.A1*(self.delta1*L1**self.alpha1 + self.beta1*K1**self.alpha1)**(1/self.alpha1),
        L1 - px1 * X1 / (w + r*(self.delta1*r/(self.beta1*w))**(1/(self.alpha1-1))),
        K1 - px1 * X1 / (r + w*(self.beta1*w /(self.delta1*r))**(1/(self.alpha1-1))),
        I11 - self.a11*Z1,
        I21 - self.a21*Z1,
        I31 - self.a31*Z1,
        Ir1 - self.ar1*Z1,
        Ib1 - self.ab1*Z1,
        X1  - self.x1*Z1,
        pz1 - (px1*self.x1 + self.a11*pq1 + self.a21*pq2 + self.a31*pq3 +  self.ar1*pqr + self.ab1*pdb),
        Z1  - self.theta1 * (self.e1*E1**self.rho1 + self.dt1*D1**self.rho1)**(1/self.rho1),
        E1  - (self.theta1 ** self.rho1 * self.e1 * (1+self.tz1 + self.tva1)*pz1 / pe1 )**(1/(1-self.rho1))*Z1,
        D1  - (self.theta1 ** self.rho1 * self.dt1 * (1+self.tz1 + self.tva1)*pz1 / pd1 )**(1/(1-self.rho1))*Z1,
        Q1  - self.lambda1*(self.m1*M1**self.eta1 + self.da1*D1**self.eta1)**(1/self.eta1),
        M1  - (self.lambda1**self.eta1 * self.m1  *pq1 / (pm1))**(1/(1-self.eta1))*Q1,
        D1  - (self.lambda1**self.eta1 * self.da1 *pq1 / pd1)**(1/(1-self.eta1))*Q1,
        X2 - self.A2*(self.delta2*L2**self.alpha2 + self.beta2*K2**self.alpha2)**(1/self.alpha2),
        L2 - px2 * X2 / (w + r*(self.delta2*r/(self.beta2*w))**(1/(self.alpha2-1))),
        K2 - px2 * X2 / (r + w*(self.beta2*w /(self.delta2*r))**(1/(self.alpha2-1))),
        I12 - self.a12*Z2,
        I22 - self.a22*Z2,
        I32 - self.a32*Z2,
        Ir2 - self.ar2*Z2,
        Ib2 - self.ab2*Z2,
        X2  - self.x2*Z2,
        pz2 - (px2*self.x2 + self.a12*pq1 + self.a22*pq2 + self.a32*pq3 + self.ar2*pqr + self.ab2*pdb),
        Z2  - self.theta2 * (self.e2*E2**self.rho2 + self.dt2*D2**self.rho2)**(1/self.rho2),
        E2  - (self.theta2 ** self.rho2 * self.e2 * (1+self.tz2 + self.tva2)*pz2 / pe2 )**(1/(1-self.rho2))*Z2,
        D2  - (self.theta2 ** self.rho2 * self.dt2 * (1+self.tz2 + self.tva2)*pz2 / pd2 )**(1/(1-self.rho2))*Z2,
        Q2  - self.lambda2*(self.m2*M2**self.eta2 + self.da2*D2**self.eta2)**(1/self.eta2),
        M2  - (self.lambda2**self.eta2 * self.m2  *pq2 / (pm2))**(1/(1-self.eta2))*Q2,
        D2  - (self.lambda2**self.eta2 * self.da2 *pq2 / pd2)**(1/(1-self.eta2))*Q2,
        X3 - self.A3*(self.delta3*L3**self.alpha3 + self.beta3*K3**self.alpha3)**(1/self.alpha3),
        L3 - px3 * X3 / (w + r*(self.delta3*r/(self.beta3*w))**(1/(self.alpha3-1))),
        K3 - px3 * X3 / (r + w*(self.beta3*w /(self.delta3*r))**(1/(self.alpha3-1))),
        I13 - self.a13*Z3,
        I23 - self.a23*Z3,
        I33 - self.a33*Z3,
        Ir3 - self.ar3*Z3,
        Ib3 - self.ab3*Z3,
        X3  - self.x3*Z3,
        pz3 - (px3*self.x3 + self.a13*pq1 + self.a23*pq2 + self.a33*pq3 + self.ar3*pqr + self.ab3*pdb),
        Z3  - self.theta3 * (self.e3*E3**self.rho3 + self.dt3*D3**self.rho3)**(1/self.rho3),
        E3  - (self.theta3 ** self.rho3 * self.e3 * (1+self.tz3 + self.tva3)*pz3 / pe3 )**(1/(1-self.rho3))*Z3,
        D3  - (self.theta3 ** self.rho3 * self.dt3 * (1+self.tz3 + self.tva3)*pz3 / pd3 )**(1/(1-self.rho3))*Z3,
        Q3  - self.lambda3*(self.m3*M3**self.eta3 + self.da3*D3**self.eta3)**(1/self.eta3),
        M3  - (self.lambda3**self.eta3 * self.m3  *pq3 / (pm3))**(1/(1-self.eta3))*Q3,
        D3  - (self.lambda3**self.eta3 * self.da3 *pq3 / pd3)**(1/(1-self.eta3))*Q3,
        Xr - self.Ar*(self.deltar*Lr**self.alphar + self.betar*Kr**self.alphar)**(1/self.alphar),
        Lr - pxr * Xr / (w + r*(self.deltar*r/(self.betar*w))**(1/(self.alphar-1))),
        Kr - pxr * Xr / (r + w*(self.betar*w /(self.deltar*r))**(1/(self.alphar-1))),
        COr  - self.lambdaco*(self.mcor*MCOr**self.etaco + self.dcor*DCOr**self.etaco)**(1/self.etaco),
        MCOr - (self.lambdaco**self.etaco*self.mcor*pco/pmco)**(1 / (1-self.etaco)) * COr,
        DCOr - (self.lambdaco**self.etaco*self.dcor*pco/pdco)**(1 / (1-self.etaco)) * COr,
        XCOr - self.Axco * (self.xr*Xr**self.alphaxco + self.co*COr**self.alphaxco)**(1/self.alphaxco),
        Xr - (self.Axco**self.alphaxco * self.xr * pxco  / pxr) ** (1 / (1-self.alphaxco)) * XCOr,
        COr - (self.Axco**self.alphaxco * self.co * pxco  / pco) ** (1 / (1-self.alphaxco)) * XCOr,
        I1r  - self.a1r*Zr,
        I2r  - self.a2r*Zr,
        I3r  - self.a3r*Zr,
        Irr  - self.arr*Zr,
        Ibr  - self.abr*Zr,
        XCOr - self.xcor*Zr,
        pzr  - (pxco*self.xcor + self.a1r*pq1 + self.a2r*pq2 + self.a3r*pq3 + self.arr*pqr + self.abr*pdb),
        Zr   - self.thetar * (self.er*Er**self.rhor + self.dtr*Dr**self.rhor)**(1/self.rhor),
        Er   - (self.thetar ** self.rhor * self.er * (1+self.tzr + self.tvar)*pzr / per )**(1/(1-self.rhor))*Zr,
        Dr   - (self.thetar ** self.rhor * self.dtr * (1+self.tzr + self.tvar)*pzr / pdr )**(1/(1-self.rhor))*Zr,
        Qr   - self.lambdar*(self.mr*Mr**self.etar + self.dar*Dr**self.etar)**(1/self.etar),
        Mr   - (self.lambdar**self.etar * self.mr  *pqr / pmr)**(1/(1-self.etar))*Qr,
        Dr   - (self.lambdar**self.etar * self.dar *pqr / pdr)**(1/(1-self.etar))*Qr,
        Xb - self.Ab*(self.deltab*Lb**self.alphab + self.betab*Kb**self.alphab)**(1/self.alphab),
        Lb - pxb * Xb / (w + r*(self.deltab*r/(self.betab*w))**(1/(self.alphab-1))),
        Kb - pxb * Xb / (r + w*(self.betab*w /(self.deltab*r))**(1/(self.alphab-1))),
        NGb  - self.lambdang * (self.mngb*MNGb**self.etang + self.dngb*DNGb**self.etang)**(1/self.etang),
        MNGb - (self.lambdang**self.etang * self.mngb*png / pmng)**(1/(1-self.etang)) * NGb,
        DNGb - (self.lambdang**self.etang * self.dngb * png / pdng)**(1/(1-self.etang)) * NGb,
        XNGb - self.Axng * (self.xb*Xb**self.alphaxng + self.ng*NGb**self.alphaxng)**(1/self.alphaxng),
        Xb - (self.Axng**self.alphaxng * self.xb * pxng  / pxb) ** (1 / (1-self.alphaxng)) * XNGb,
        NGb - (self.Axng**self.alphaxng * self.ng * pxng  / png) ** (1 / (1-self.alphaxng)) * XNGb,
        I1b  - self.a1b*Zb,
        I2b  - self.a2b*Zb,
        I3b  - self.a3b*Zb,
        Irb  - self.arb*Zb,
        Ibb  - self.abb*Zb,
        XNGb - self.xngb*Zb,
        pzb  - (pxng*self.xngb + self.a1b*pq1 + self.a2b*pq2 + self.a3b*pq3 + self.arb*pqr + self.abb*pdb),
        Zb   - self.thetab * (self.eb*Eb**self.rhob + self.dtb*Db**self.rhob)**(1/self.rhob),
        Eb   - (self.thetab ** self.rhob * self.eb  * (1+self.tzb + self.tvab) *pzb / peb )**(1/(1-self.rhob))*Zb,
        Db   - (self.thetab ** self.rhob * self.dtb * (1+self.tzb + self.tvab) *pzb / pdb )**(1/(1-self.rhob))*Zb,
        C1 - self.c1 / pq1 * Yd,
        C2 - self.c2 / pq2 * Yd,
        C3 - self.c3 / pq3 * Yd,
        Cr - self.cr / pqr * Yd,
        Cb - self.cb / pdb * Yd,
        Yd - (Y - Sp - Td),
        Y  - (w*self.Lbar + r*self.Kbar),
        TPAO1 - self.mu1 / pq1 * OIL_INCOME,
        TPAO2 - self.mu2 / pq2 * OIL_INCOME,
        TPAO3 - self.mu3 / pq3 * OIL_INCOME,
        OIL_INCOME - (pdco * self.DCOBar + pdng*self.DNGBar + self.epsilon*self.E_Energy),
        # OIL_INCOME - (pdco * self.DCOBar + pdng*self.DNGBar + (self.Pwmco + self.Pwmng)/2*self.E_Energy),
        G1   - self.g1 / pq1 * (T-Sg),
        G2   - self.g2 / pq2 * (T-Sg),
        G3   - self.g3 / pq3 * (T-Sg),
        T    - (Td + Tz + Tva),
        Td   - self.td*Y,
        Tz   - (Tz1 + Tz2 + Tz3 + Tzr + Tzb),
        Tz1  - self.tz1 * pz1 * Z1,
        Tz2  - self.tz2 * pz2 * Z2,
        Tz3  - self.tz3 * pz3 * Z3,
        Tzr  - self.tzr * pzr * Zr,
        Tzb  - self.tzb * pzb * Zb,
        Tva  - (Tva1 + Tva2 + Tva3 +  Tvar + Tvab), 
        Tva1 - self.tva1 * pz1 * Z1,
        Tva2 - self.tva2 * pz2 * Z2,
        Tva3 - self.tva3 * pz3 * Z3,
        Tvar - self.tvar * pzr * Zr,
        Tvab - self.tvab * pzb * Zb,
        INV1 - self.inv1 / pq1 * S,
        INV2 - self.inv2 / pq2 * S,
        INV3 - self.inv3 / pq3 * S,
        S    - (Sp + Sg + Sf*self.epsilon),
        Sp   - self.sp*Y,
        Sg   - self.sg*T,
        pe1 - self.epsilon * self.Pwe1,
        pe2 - self.epsilon * self.Pwe2,
        pe3 - self.epsilon * self.Pwe3,
        per - self.epsilon * self.Pwer,
        peb - self.epsilon * self.Pweb,
        pm1 - self.epsilon * self.Pwm1,
        pm2 - self.epsilon * self.Pwm2,
        pm3 - self.epsilon * self.Pwm3,
        pmr - self.epsilon * self.Pwmr,
        pmco - self.epsilon * self.Pwmco,
        pmng - self.epsilon * self.Pwmng,
        self.Pwe1*E1 + self.Pwe2*E2 + self.Pwe3*E3 +  self.Pwer*Er + self.Pweb*Eb + Sf + self.E_Energy - (self.Pwm1*M1 + self.Pwm2*M2 + self.Pwm3*M3 +  self.Pwmr*Mr + self.Pwmco * MCOr + self.Pwmng * MNGb),
        Q1 - (C1 + TPAO1 + G1 + INV1 + I11 + I12 + I13 +  I1r + I1b),
        Q2 - (C2 + TPAO2 + G2 + INV2 + I21 + I22 + I23 +  I2r + I2b),
        # Q3 - (C3 + TPAO3 + G3 + INV3 + I31 + I32 + I33 +  I3r + I3b),
        Qr - (Cr + Ir1 + Ir2 + Ir3 + Irr + Irb),
        Db - (Cb + Ib1 + Ib2 + Ib3 + Ibr + Ibb),
        self.Lbar - (L1 + L2 + L3 + Lr + Lb),
        self.Kbar - (K1 + K2 + K3 + Kr + Kb),
        self.DCOBar - DCOr,
        self.DNGBar - DNGb,

        ]
    
    def SolveModel(self):
        
        cons = {"type":"eq", "fun":self.constraints}
        x0 = self.init_values
        bnds = []

        for val in self.init_values_str:
            if val == "Tz1" or val == "Tva1":
                bnds.append((None, None))
            else:
                bnds.append((0.000000000001, None))
        
        result = minimize(self.objValue,
                          x0, 
                          constraints = cons,
                         bounds = bnds) 

        return result
        
# SAM = pd.read_excel("SHMCFrun.xlsx", index_col = "index")
# model1 = CGE(SAM)
# result1 = model1.SolveModel()

# model2 = CGE(SAM)
# model2.Pwmco*=1.3
# model2.Pwmng*=1.3

# result2 = model2.SolveModel()


# print(result1.message)
# print(result2.message)

# for i in result2.x:
#     print(i)