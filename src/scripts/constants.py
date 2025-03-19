"""
*******************************************************************************
Copyright (C) [2025] [ATTOP project]

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License, version 3,
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License, version 3,
along with this program. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************

*******************************************************************************
Constants used in other scripts
*******************************************************************************
"""

# Dictionary of most elements matched with atomic masses
ELEMENTS_ATOMIC_MASS = dict(
    H=1.008, HE=4.003, LI=6.941, BE=9.012, B=10.811, C=12.011, N=14.007, O=15.999, F=18.998, NE=20.180,
    NA=22.990, MG=24.305, AL=26.982, SI=28.086, P=30.974, S=32.066, CL=35.453, AR=39.948, K=39.098,
    CA=40.078, SC=44.956, TI=47.867, V=50.942, CR=51.996, MN=54.938, FE=55.845, CO=58.933, NI=58.693,
    CU=63.546, ZN=65.38, GA=69.723, GE=72.631, AS=74.922, SE=78.971, BR=79.904, KR=84.798, RB=84.468,
    SR=87.62, Y=88.906, ZR=91.224, NB=92.906, MO=95.95, TC=98.907, RU=101.07, RH=102.906, PD=106.42,
    AG=107.868, CD=112.414, IN=114.818, SN=118.711, SB=121.760, TE=126.7, I=126.904, XE=131.294,
    CS=132.905, BA=137.328, LA=138.905, CE=140.116, PR=140.908, ND=144.243, PM=144.913, SM=150.36,
    EU=151.964, GD=157.25, TB=158.925, DY=162.500, HO=164.930, ER=167.259, TM=168.934, YB=173.055,
    LU=174.967, HF=178.49, TA=180.948, W=183.84, RE=186.207, OS=190.23, IR=192.217, PT=195.085,
    AU=196.967, HG=200.592, TL=204.383, PB=207.2, BI=208.980, PO=208.982, AT=209.987, RN=222.081,
    FR=223.020, RA=226.025, AC=227.028, TH=232.038, PA=231.036, U=238.029, NP=237, PU=244, AM=243,
    CM=247, BK=247, CT=251, ES=252, FM=257, MD=258, NO=259, LR=262, RF=261, DB=262, SG=266, BH=264,
    HS=269, MT=268, DS=271, RG=272, CN=285, NH=284, FL=289, MC=288, LV=292, TS=294, OG=294,
    He=4.003, Li=6.941, Be=9.012, Ne=20.180,
    Na=22.990, Mg=24.305, Al=26.982, Si=28.086, Cl=35.453, Ar=39.948,
    Ca=40.078, Sc=44.956, Ti=47.867, Cr=51.996, Mn=54.938, Fe=55.845, Co=58.933, Ni=58.693,
    Cu=63.546, Zn=65.38, Ga=69.723, Ge=72.631, As=74.922, Se=78.971, Br=79.904, Kr=84.798, Rb=84.468,
    Sr=87.62, Zr=91.224, Nb=92.906, Mo=95.95, Tc=98.907, Ru=101.07, Rh=102.906, Pd=106.42,
    Ag=107.868, Cd=112.414, In=114.818, Sn=118.711, Sb=121.760, Te=126.7, Xe=131.294,
    Cs=132.905, Ba=137.328, La=138.905, Ce=140.116, Pr=140.908, Nd=144.243, Pm=144.913, Sm=150.36,
    Eu=151.964, Gd=157.25, Tb=158.925, Dy=162.500, Ho=164.930, Er=167.259, Tm=168.934, Yb=173.055,
    Lu=174.967, Hf=178.49, Ta=180.948, Re=186.207, Os=190.23, Ir=192.217, Pt=195.085,
    Au=196.967, Hg=200.592, Tl=204.383, Pb=207.2, Bi=208.980, Po=208.982, At=209.987, Rn=222.081,
    Fr=223.020, Ra=226.025, Ac=227.028, Th=232.038, Pa=231.036, Np=237, Pu=244, Am=243,
    Cm=247, Bk=247, Ct=251, Es=252, Fm=257, Md=258, No=259, Lr=262, Rf=261, Db=262, Sg=266, Bh=264,
    Hs=269, Mt=268, Ds=271, Rg=272, Cn=285, Nh=284, Fl=289, Mc=288, Lv=292, Ts=294, Og=294
)

BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1 / BOHR_TO_ANG
AU_TO_FS = 0.02418884254                # a.u. of time to femtoseconds
HARTREE_TO_EV = 27.2113966413     
INV_CM_TO_HARTREE = 4.556335*10**-6     # cm-1 to a.u. (Hartree)
AMU_TO_ME = 1822.888                    # Atomic Mass Unit to electron's mass (a.u. of mass)
