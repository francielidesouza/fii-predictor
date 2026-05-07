"""
corrigir_dataset.py
-------------------
Corrige o dataset dataset_fiis_2019_2024_brapi.xlsx:

1. Exclui ETFs
2. Remapeia fundos do segmento Multicategoria para seus segmentos reais
3. Corrige classificacoes incorretas identificadas via investidor10.com.br
4. Padroniza nomes de Segmento e Tipo_do_Fundo (sem acentos, sem variacoes)

Fonte das classificacoes: investidor10.com.br

Uso:
    python corrigir_dataset.py
"""

import pandas as pd

ENTRADA = "dataset_fiis_2019_2024_brapi.xlsx"
SAIDA   = "dataset_fiis_2019_2024_brapi_v2.xlsx"

# ── ETFs — excluir completamente ─────────────────────────────────────────────
EXCLUIR = {"GLDN11", "SCOO11"}

# ── Remapeamento completo — fonte: investidor10.com.br ───────────────────────
# Formato: "SIGLA": ("Segmento", "Tipo_do_Fundo")
REMAPEAR = {
    # Multicategoria → corrigidos
    "ALZR11": ("Hibrido",            "Tijolo"),
    "BCIA11": ("Titulos e Val. Mob.","FOF"),
    "BRCR11": ("Hibrido",            "Tijolo"),
    "BRRI11": ("Hibrido",            "Tijolo"),
    "BTLG11": ("Logistico",          "Tijolo"),
    "CRFF11": ("Titulos e Val. Mob.","FOF"),
    "CXRI11": ("Hibrido",            "FOF"),
    "EDGA11": ("Lajes Corporativas", "Tijolo"),
    "FCFL11": ("Educacional",        "Tijolo"),
    "FPNG11": ("Lajes Corporativas", "Tijolo"),
    "HFOF11": ("Titulos e Val. Mob.","FOF"),
    "HGLG11": ("Logistico",          "Tijolo"),
    "JSRE11": ("Hibrido",            "Outros"),
    "KFOF11": ("Titulos e Val. Mob.","FOF"),
    "KNCR11": ("Titulos e Val. Mob.","Papel"),
    "KNHY11": ("Titulos e Val. Mob.","Papel"),
    "KNIP11": ("Titulos e Val. Mob.","Papel"),
    "KNRI11": ("Hibrido",            "Tijolo"),
    "MAXR11": ("Varejo",             "Outros"),
    "MCCI11": ("Titulos e Val. Mob.","Papel"),
    "MFII11": ("Hibrido",            "Desenvolvimento"),
    "MVFI11": ("Hibrido",            "Misto"),
    "NCHB11": ("Titulos e Val. Mob.","Papel"),
    "OUJP11": ("Titulos e Val. Mob.","Papel"),
    "PATB11": ("Hibrido",            "Tijolo"),
    "PLRI11": ("Titulos e Val. Mob.","Papel"),
    "PORD11": ("Titulos e Val. Mob.","Papel"),
    "RBRD11": ("Hibrido",            "Tijolo"),
    "RBRR11": ("Titulos e Val. Mob.","Papel"),
    "RECT11": ("Hibrido",            "Outros"),
    "SADI11": ("Titulos e Val. Mob.","Papel"),
    "TGAR11": ("Hibrido",            "Desenvolvimento"),
    "VERE11": ("Hibrido",            "Tijolo"),
    "VGIR11": ("Titulos e Val. Mob.","Papel"),
    "VOTS11": ("Titulos e Val. Mob.","Papel"),
    "XPML11": ("Shoppings",          "Tijolo"),
    # Correcoes adicionais — investidor10.com.br
    "MXRF11": ("Hibrido",            "Papel"),
    "CVBI11": ("Titulos e Val. Mob.","Papel"),
    "FEXC11": ("Titulos e Val. Mob.","Papel"),
    "HGCR11": ("Titulos e Val. Mob.","Papel"),
    "IRDM11": ("Titulos e Val. Mob.","Papel"),
    "VCJR11": ("Titulos e Val. Mob.","Papel"),
    "VRTA11": ("Titulos e Val. Mob.","Papel"),
    "VTLT11": ("Logistico",          "Tijolo"),
}

# ── Padronizacao de Segmento ─────────────────────────────────────────────────
PADRONIZAR_SEG = {
    # Com acento → sem acento
    "Títulos e Val. Mob.":            "Titulos e Val. Mob.",
    "Logística":                      "Logistico",
    "Logístico":                      "Logistico",
    "Híbrido":                        "Hibrido",
    "Híbrida":                        "Hibrido",
    "Escritórios":                    "Escritorios",
    # Variacoes de nome
    "Titulos de Valores Imobiliarios":"Titulos e Val. Mob.",
    "Titulos e Valores Imobiliarios": "Titulos e Val. Mob.",
    "Shopping":                       "Shoppings",
    "Lajes Corporativas":             "Lajes Corporativas",
    "Shoppings":                      "Shoppings",
    "Hospital":                       "Hospital",
    "Residencial":                    "Residencial",
    "Educacional":                    "Educacional",
    "Varejo":                         "Varejo",
    "Outros":                         "Outros",
    "outros":                         "Outros",
    "Multicategoria":                 "Multicategoria",
}

# ── Padronizacao de Tipo_do_Fundo ────────────────────────────────────────────
PADRONIZAR_TIPO = {
    "papel":           "Papel",
    "Papel":           "Papel",
    "tijolo":          "Tijolo",
    "Tijolo":          "Tijolo",
    "fof":             "FOF",
    "FOF":             "FOF",
    "hibrido":         "Hibrido",
    "híbrido":         "Hibrido",
    "Híbrido":         "Hibrido",
    "Hibrido":         "Hibrido",
    "outros":          "Outros",
    "Outros":          "Outros",
    "desenvolvimento": "Desenvolvimento",
    "Desenvolvimento": "Desenvolvimento",
    "misto":           "Misto",
    "Misto":           "Misto",
}


def corrigir():
    print(f"\n{'='*58}")
    print(f"  Corrigindo dataset — versao final")
    print(f"  Fonte: investidor10.com.br")
    print(f"{'='*58}\n")

    df = pd.read_excel(ENTRADA)
    print(f"[✓] Carregado: {len(df)} linhas · {df['Sigla'].nunique()} fundos")

    # 1. Exclui ETFs
    antes = df['Sigla'].nunique()
    df = df[~df['Sigla'].isin(EXCLUIR)]
    excluidos = antes - df['Sigla'].nunique()
    print(f"[✓] ETFs excluidos: {excluidos} fundos {EXCLUIR}")

    # Remove fundos sem Tipo_do_Fundo definido
    antes_nan = df['Sigla'].nunique()
    df = df[df['Tipo_do_Fundo'].notna()]
    df = df[df['Tipo_do_Fundo'].str.strip() != '']
    nan_excluidos = antes_nan - df['Sigla'].nunique()
    print(f"[✓] Fundos sem Tipo_do_Fundo excluidos: {nan_excluidos}")

    # 2. Aplica remapeamento
    remapeados = 0
    for sigla, (seg_novo, tipo_novo) in REMAPEAR.items():
        mask = df['Sigla'] == sigla
        if mask.sum() > 0:
            seg_ant  = df.loc[mask, 'Segmento'].iloc[0]
            tipo_ant = df.loc[mask, 'Tipo_do_Fundo'].iloc[0]
            df.loc[mask, 'Segmento']      = seg_novo
            df.loc[mask, 'Tipo_do_Fundo'] = tipo_novo
            if seg_ant != seg_novo or tipo_ant != tipo_novo:
                print(f"  {sigla}: {seg_ant}/{tipo_ant} → {seg_novo}/{tipo_novo}")
                remapeados += 1
    print(f"\n[✓] {remapeados} fundos remapeados")

    # 3. Padroniza Segmento
    df['Segmento'] = df['Segmento'].map(
        lambda x: PADRONIZAR_SEG.get(str(x), x) if pd.notna(x) else x
    )
    print(f"[✓] Segmentos padronizados")

    # 4. Padroniza Tipo_do_Fundo
    df['Tipo_do_Fundo'] = df['Tipo_do_Fundo'].map(
        lambda x: PADRONIZAR_TIPO.get(str(x), x) if pd.notna(x) else x
    )
    print(f"[✓] Tipos padronizados")

    # 5. Verifica Multicategoria restante
    mc = df[df['Segmento'] == 'Multicategoria']['Sigla'].unique()
    if len(mc) > 0:
        print(f"\n⚠  Ainda em Multicategoria ({len(mc)}): {list(mc)}")
    else:
        print(f"[✓] Nenhum fundo em Multicategoria!")

    # 6. Resumo final
    print(f"\n{'─'*58}")
    print(f"Dataset final: {len(df)} linhas · {df['Sigla'].nunique()} fundos")
    print(f"\nDistribuicao por Tipo + Segmento:")
    dist = df.groupby(['Tipo_do_Fundo','Segmento'])['Sigla'].nunique()
    print(dist.to_string())

    # 7. Salva
    df.to_excel(SAIDA, index=False)
    print(f"\n[✓] Salvo: {SAIDA}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    corrigir()