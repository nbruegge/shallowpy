# --- cut masks
maskup = create_output_var(masku, dims=dimsu)
maskvp = create_output_var(maskv, dims=dimsv)
masktp = create_output_var(maskt, dims=dimst)

# --- cut variables
uop = create_output_var(uo, dims=dimsu)
uop = uop.where(maskup==1)
vop = create_output_var(vo, dims=dimsv)
vop = vop.where(maskvp==1)
hop = create_output_var(ho, dims=dimst)
hop = hop.where(masktp==1)

# --- combine tendencies to dictionary
Tuo = dict()
Tuo['tot'] = Tuo_tot[:,nspy:-nspy, nspx:-nspx]
Tuo['adv'] = Tuo_adv[:,nspy:-nspy, nspx:-nspx]
Tuo['dif'] = Tuo_dif[:,nspy:-nspy, nspx:-nspx]
Tuo['pgd'] = Tuo_pgr[:,nspy:-nspy, nspx:-nspx]
Tuo['cor'] = Tuo_cor[:,nspy:-nspy, nspx:-nspx]
Tuo['vdf'] = Tuo_vdf[:,nspy:-nspy, nspx:-nspx]
Tvo = dict()
Tvo['tot'] = Tvo_tot[:,nspy:-nspy, nspx:-nspx]
Tvo['adv'] = Tvo_adv[:,nspy:-nspy, nspx:-nspx]
Tvo['dif'] = Tvo_dif[:,nspy:-nspy, nspx:-nspx]
Tvo['pgd'] = Tvo_pgr[:,nspy:-nspy, nspx:-nspx]
Tvo['cor'] = Tvo_cor[:,nspy:-nspy, nspx:-nspx]
Tvo['vdf'] = Tvo_vdf[:,nspy:-nspy, nspx:-nspx]
Tho = dict()
Tho['tot'] = Tho_tot[:,nspy:-nspy, nspx:-nspx]
Tho['adv'] = Tho_adv[:,nspy:-nspy, nspx:-nspx]
Tho['dif'] = Tho_dif[:,nspy:-nspy, nspx:-nspx]
Tho['mix'] = Tho_mix[:,nspy:-nspy, nspx:-nspx]
Tho['con'] = Tho_con[:,nspy:-nspy, nspx:-nspx]
