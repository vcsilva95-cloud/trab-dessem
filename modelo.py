"""modelo.py - Programacao semanal hidrotermica MILP com PySCIPOpt.

Sistema academico: UHE Chavantes + UTE Termobahia UG1 + UTE Campina Grande UG1.
Horizonte: 168 h, discretizado em 336 intervalos de 30 min (48 por dia).

Observacao importante:
- A curva de carga default e sintetica e normalizada para media semanal de 500 MW.
- Custos default sao parametros academicos editaveis, nao dados cadastrais oficiais.
- O PySDDP/PDE e disponibilizado por uma funcao adaptadora para leitura opcional
  de um caso DESSEM. A otimizacao MILP deste arquivo e montada no PySCIPOpt.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum

# Compatibilidade entre publicacoes/namespaces do pacote PySDDP.
# Mantem literalmente o import solicitado para ambientes que o exponham assim.
try:
    from pysddp import pde  # type: ignore
except ImportError:
    try:
        from PySDDP import Pde as pde  # type: ignore
    except ImportError:
        pde = None


DT_H = 0.5
N_DIAS = 7
PATAMARES_DIA = 48
N_PERIODOS = N_DIAS * PATAMARES_DIA


@dataclass
class Termica:
    nome: str
    pmin_mw: float
    pmax_mw: float
    ton_h: float
    toff_h: float
    rampa_subida_mw_h: float
    rampa_descida_mw_h: float
    cvu_rs_mwh: float
    custo_partida_rs: float = 0.0
    custo_parada_rs: float = 0.0
    custo_fixo_rs_h: float = 0.0
    trajetoria_partida_mw: List[float] = field(default_factory=list)
    trajetoria_parada_mw: List[float] = field(default_factory=list)
    ligada_inicial: int = 0
    geracao_inicial_mw: float = 0.0
    tempo_estado_inicial_h: float = 24.0


@dataclass
class Hidreletrica:
    nome: str = "UHE Chavantes"
    volume_inicial_hm3: float = 7300.0  # principal entrada da futura interface
    volume_min_hm3: float = 5754.0
    volume_max_hm3: float = 8795.0
    vazao_afluente_m3s: float = 350.0  # constante em toda a semana
    vazao_turbinada_max_m3s: float = 1761.7
    produtibilidade_mw_por_m3s: float = 0.235
    geracao_max_mw: float = 414.0
    vazao_min_m3s: float = 0.0
    volume_meta_final_hm3: Optional[float] = None
    penalidade_vertimento_rs_hm3: float = 0.01
    penalidade_volume_final_rs_hm3: float = 500.0


@dataclass
class Sistema:
    carga_mw: List[float]
    termicas: List[Termica]
    uhe: Hidreletrica
    custo_deficit_rs_mwh: float = 10000.0
    reserva_percentual: float = 0.0
    limite_tempo_s: Optional[float] = None
    gap_relativo: float = 0.001
    exibir_log_solver: bool = True


def curva_carga_semanal(media_mw: float = 500.0) -> List[float]:
    """Gera perfil sintetico de 336 pontos e o normaliza para a media informada.

    O formato tem vale na madrugada, depressao diurna associada a MMGD e ponta
    noturna. Sabado e domingo recebem fatores menores que os dias uteis.
    """
    valores: List[float] = []
    for dia in range(7):
        fator_dia = 1.0 if dia < 5 else (0.92 if dia == 5 else 0.86)
        for k in range(PATAMARES_DIA):
            h = k * DT_H
            base = 0.98
            madrugada = -0.13 * math.exp(-0.5 * ((h - 4.0) / 2.6) ** 2)
            vale_solar = -0.12 * math.exp(-0.5 * ((h - 13.0) / 2.5) ** 2)
            ponta = 0.22 * math.exp(-0.5 * ((h - 19.5) / 2.0) ** 2)
            manha = 0.06 * math.exp(-0.5 * ((h - 9.0) / 2.0) ** 2)
            valores.append(fator_dia * (base + madrugada + vale_solar + ponta + manha))
    escala = media_mw / float(np.mean(valores))
    return [round(v * escala, 3) for v in valores]


def parametros_padrao() -> Sistema:
    """Parametros centrais, prontos para serem sobrescritos pela interface."""
    termobahia = Termica(
        nome="Termobahia UG1", pmin_mw=60.68, pmax_mw=148.00,
        ton_h=6.0, toff_h=4.0,
        rampa_subida_mw_h=166.0,
        # Hipotese editavel: valor de descida igual ao de subida, pois a tabela
        # fornecida nao identificou o campo correspondente para Termobahia.
        rampa_descida_mw_h=166.0,
        cvu_rs_mwh=350.0, custo_partida_rs=12000.0,
        custo_parada_rs=1000.0, custo_fixo_rs_h=0.0,
        trajetoria_partida_mw=[2.0, 60.68],
        trajetoria_parada_mw=[0.0],
    )
    campina = Termica(
        nome="Campina Grande UG1", pmin_mw=8.45, pmax_mw=42.27,
        ton_h=8.0, toff_h=8.0,
        rampa_subida_mw_h=34.0, rampa_descida_mw_h=34.0,
        cvu_rs_mwh=800.0, custo_partida_rs=5000.0,
        custo_parada_rs=500.0, custo_fixo_rs_h=0.0,
        trajetoria_partida_mw=[8.45],
        trajetoria_parada_mw=[0.0],
    )
    return Sistema(
        carga_mw=curva_carga_semanal(500.0),
        termicas=[termobahia, campina],
        uhe=Hidreletrica(),
    )


def aplicar_entradas(base: Sistema, entradas: Dict[str, Any]) -> Sistema:
    """Atualiza o objeto Sistema a partir de um dicionario/JSON da interface."""
    dados = asdict(base)
    for chave in ("carga_mw", "custo_deficit_rs_mwh", "reserva_percentual",
                  "limite_tempo_s", "gap_relativo", "exibir_log_solver"):
        if chave in entradas:
            dados[chave] = entradas[chave]
    if "uhe" in entradas:
        dados["uhe"].update(entradas["uhe"])
    if "termicas" in entradas:
        por_nome = {t["nome"]: t for t in dados["termicas"]}
        for atualizacao in entradas["termicas"]:
            nome = atualizacao["nome"]
            if nome not in por_nome:
                raise ValueError(f"Termica desconhecida: {nome}")
            por_nome[nome].update(atualizacao)
        dados["termicas"] = list(por_nome.values())
    return Sistema(
        carga_mw=list(map(float, dados["carga_mw"])),
        termicas=[Termica(**t) for t in dados["termicas"]],
        uhe=Hidreletrica(**dados["uhe"]),
        custo_deficit_rs_mwh=float(dados["custo_deficit_rs_mwh"]),
        reserva_percentual=float(dados["reserva_percentual"]),
        limite_tempo_s=dados["limite_tempo_s"],
        gap_relativo=float(dados["gap_relativo"]),
        exibir_log_solver=bool(dados["exibir_log_solver"]),
    )


def validar(s: Sistema) -> None:
    if len(s.carga_mw) != N_PERIODOS:
        raise ValueError(f"carga_mw deve ter {N_PERIODOS} valores de meia hora.")
    if min(s.carga_mw) < 0:
        raise ValueError("A carga nao pode ser negativa.")
    if not (s.uhe.volume_min_hm3 <= s.uhe.volume_inicial_hm3 <= s.uhe.volume_max_hm3):
        raise ValueError("Volume inicial da UHE fora dos limites.")
    for g in s.termicas:
        if not (0 <= g.pmin_mw <= g.pmax_mw):
            raise ValueError(f"Limites invalidos para {g.nome}.")
        if g.ton_h < 0 or g.toff_h < 0:
            raise ValueError(f"TON/TOFF invalidos para {g.nome}.")
        if g.trajetoria_partida_mw and abs(g.trajetoria_partida_mw[-1] - g.pmin_mw) > 1e-6:
            raise ValueError(f"A trajetoria de partida de {g.nome} deve terminar em Pmin.")


def carregar_caso_dessem_pysddp(caminho: str, arquivo_mestre: str = "dessem.arq") -> Any:
    """Leitura opcional de um caso DESSEM pelo modulo PDE do PySDDP."""
    if pde is None:
        raise ImportError("PySDDP nao instalado. Instale a versao compativel do pacote.")
    classe = getattr(pde, "Dessem", None)
    if classe is None:
        raise AttributeError("O modulo pde/Pde instalado nao expoe a classe Dessem.")
    return classe(caminho, arquivo_mestre)


def _soma_eventos(var, inicio: int, fim: int):
    indices = range(max(0, inicio), min(N_PERIODOS, fim))
    return quicksum(var[k] for k in indices)


def construir_modelo(s: Sistema):
    validar(s)
    m = Model("PDO_hidrotermica_7_dias")
    if not s.exibir_log_solver:
        m.hideOutput()
    m.setRealParam("limits/gap", s.gap_relativo)
    if s.limite_tempo_s is not None:
        m.setRealParam("limits/time", float(s.limite_tempo_s))

    T = range(N_PERIODOS)
    G = range(len(s.termicas))

    u = {(i,t): m.addVar(vtype="B", name=f"u_{i}_{t}") for i in G for t in T}
    y = {(i,t): m.addVar(vtype="B", name=f"partida_{i}_{t}") for i in G for t in T}
    z = {(i,t): m.addVar(vtype="B", name=f"parada_{i}_{t}") for i in G for t in T}
    p_est = {(i,t): m.addVar(lb=0.0, name=f"p_estavel_{i}_{t}") for i in G for t in T}
    gterm = {(i,t): m.addVar(lb=0.0, name=f"gterm_{i}_{t}") for i in G for t in T}

    gh = {t: m.addVar(lb=0.0, ub=s.uhe.geracao_max_mw, name=f"gh_{t}") for t in T}
    qt = {t: m.addVar(lb=s.uhe.vazao_min_m3s, ub=s.uhe.vazao_turbinada_max_m3s, name=f"qt_{t}") for t in T}
    vert = {t: m.addVar(lb=0.0, name=f"vert_hm3_{t}") for t in T}
    vol = {t: m.addVar(lb=s.uhe.volume_min_hm3, ub=s.uhe.volume_max_hm3, name=f"vol_{t}") for t in T}
    deficit = {t: m.addVar(lb=0.0, name=f"deficit_{t}") for t in T}
    excesso = {t: m.addVar(lb=0.0, name=f"excesso_{t}") for t in T}
    desvio_meta = m.addVar(lb=0.0, name="desvio_volume_final")

    # Unit commitment, trajetorias e limites estaveis.
    for i, usina in enumerate(s.termicas):
        nup = len(usina.trajetoria_partida_mw)
        ndn = len(usina.trajetoria_parada_mw)
        ton = int(math.ceil(usina.ton_h / DT_H))
        toff = int(math.ceil(usina.toff_h / DT_H))
        for t in T:
            u_ant = usina.ligada_inicial if t == 0 else u[i,t-1]
            m.addCons(u[i,t] - u_ant == y[i,t] - z[i,t], name=f"transicao_{i}_{t}")
            m.addCons(y[i,t] + z[i,t] <= 1, name=f"evento_unico_{i}_{t}")

            # TON e TOFF em numero de intervalos de 30 minutos.
            m.addCons(_soma_eventos({k: y[i,k] for k in T}, t-ton+1, t+1) <= u[i,t],
                      name=f"ton_{i}_{t}")
            m.addCons(_soma_eventos({k: z[i,k] for k in T}, t-toff+1, t+1) <= 1-u[i,t],
                      name=f"toff_{i}_{t}")

            em_partida = _soma_eventos({k: y[i,k] for k in T}, t-nup+1, t+1) if nup else 0
            # Antes de uma parada futura, a unidade nao opera no bloco estavel.
            em_parada = _soma_eventos({k: z[i,k] for k in T}, t, t+ndn) if ndn else 0
            livre = u[i,t] - em_partida - em_parada
            m.addCons(livre >= 0, name=f"livre_nao_neg_{i}_{t}")
            m.addCons(p_est[i,t] >= usina.pmin_mw * livre, name=f"pmin_{i}_{t}")
            m.addCons(p_est[i,t] <= usina.pmax_mw * livre, name=f"pmax_{i}_{t}")

            parcela_partida = quicksum(
                usina.trajetoria_partida_mw[k] * y[i,t-k]
                for k in range(nup) if t-k >= 0
            )
            parcela_parada = quicksum(
                usina.trajetoria_parada_mw[k] * z[i,t+k]
                for k in range(ndn) if t+k < N_PERIODOS
            )
            m.addCons(gterm[i,t] == p_est[i,t] + parcela_partida + parcela_parada,
                      name=f"trajetoria_{i}_{t}")

            # Rampas fora das trajetorias; big-M libera os periodos de evento.
            g_ant = usina.geracao_inicial_mw if t == 0 else gterm[i,t-1]
            evento = y[i,t] + z[i,t]
            if t > 0:
                evento += y[i,t-1] + z[i,t-1]
            M = usina.pmax_mw
            m.addCons(gterm[i,t] - g_ant <= usina.rampa_subida_mw_h*DT_H + M*evento,
                      name=f"rampa_up_{i}_{t}")
            m.addCons(g_ant - gterm[i,t] <= usina.rampa_descida_mw_h*DT_H + M*evento,
                      name=f"rampa_down_{i}_{t}")

    # UHE: afluencia constante; geracao linear em vazao turbinada.
    fator_hm3 = DT_H * 3600.0 / 1_000_000.0
    for t in T:
        m.addCons(gh[t] == s.uhe.produtibilidade_mw_por_m3s * qt[t], name=f"producao_uhe_{t}")
        vol_ant = s.uhe.volume_inicial_hm3 if t == 0 else vol[t-1]
        m.addCons(
            vol[t] == vol_ant + s.uhe.vazao_afluente_m3s*fator_hm3
                      - qt[t]*fator_hm3 - vert[t],
            name=f"balanco_hidrico_{t}",
        )
        m.addCons(
            gh[t] + quicksum(gterm[i,t] for i in G) + deficit[t] - excesso[t] == s.carga_mw[t],
            name=f"atendimento_{t}",
        )
        if s.reserva_percentual > 0:
            m.addCons(
                s.uhe.geracao_max_mw + quicksum(s.termicas[i].pmax_mw*u[i,t] for i in G)
                >= (1.0+s.reserva_percentual)*s.carga_mw[t],
                name=f"reserva_{t}",
            )

    meta = s.uhe.volume_meta_final_hm3
    if meta is None:
        meta = s.uhe.volume_inicial_hm3
    m.addCons(desvio_meta >= meta - vol[N_PERIODOS-1], name="meta_volume_final")

    custo = quicksum(
        DT_H * s.termicas[i].cvu_rs_mwh * gterm[i,t]
        + s.termicas[i].custo_partida_rs * y[i,t]
        + s.termicas[i].custo_parada_rs * z[i,t]
        + DT_H * s.termicas[i].custo_fixo_rs_h * u[i,t]
        for i in G for t in T
    )
    custo += quicksum(
        DT_H*s.custo_deficit_rs_mwh*deficit[t]
        + s.uhe.penalidade_vertimento_rs_hm3*vert[t]
        + 1.0*excesso[t]
        for t in T
    )
    custo += s.uhe.penalidade_volume_final_rs_hm3 * desvio_meta
    m.setObjective(custo, "minimize")

    vars_ = dict(u=u, y=y, z=z, gterm=gterm, gh=gh, qt=qt, vert=vert,
                 vol=vol, deficit=deficit, excesso=excesso, desvio_meta=desvio_meta)
    return m, vars_


def resolver(s: Optional[Sistema] = None) -> Dict[str, Any]:
    if s is None:
        s = parametros_padrao()
    m, v = construir_modelo(s)
    m.optimize()
    status = str(m.getStatus())
    if m.getNSols() == 0:
        return {"status": status, "mensagem": "Nenhuma solucao incumbente encontrada."}

    linhas = []
    for t in range(N_PERIODOS):
        linha = {
            "periodo": t,
            "dia": t // PATAMARES_DIA + 1,
            "patamar_dia": t % PATAMARES_DIA + 1,
            "hora": (t % PATAMARES_DIA) * DT_H,
            "carga_mw": s.carga_mw[t],
            "uhe_mw": m.getVal(v["gh"][t]),
            "vazao_turbinada_m3s": m.getVal(v["qt"][t]),
            "volume_hm3": m.getVal(v["vol"][t]),
            "vertimento_hm3": m.getVal(v["vert"][t]),
            "deficit_mw": m.getVal(v["deficit"][t]),
            "excesso_mw": m.getVal(v["excesso"][t]),
        }
        for i, ute in enumerate(s.termicas):
            prefixo = ute.nome.lower().replace(" ", "_")
            linha[f"{prefixo}_mw"] = m.getVal(v["gterm"][i,t])
            linha[f"{prefixo}_ligada"] = round(m.getVal(v["u"][i,t]))
            linha[f"{prefixo}_partida"] = round(m.getVal(v["y"][i,t]))
            linha[f"{prefixo}_parada"] = round(m.getVal(v["z"][i,t]))
        linhas.append(linha)
    df = pd.DataFrame(linhas)
    return {
        "status": status,
        "custo_total_rs": float(m.getObjVal()),
        "gap": float(m.getGap()),
        "volume_final_hm3": float(df.iloc[-1]["volume_hm3"]),
        "resultados": df,
    }


def salvar_entradas(s: Sistema, caminho: str = "entradas.json") -> None:
    Path(caminho).write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_entradas(caminho: str = "entradas.json") -> Sistema:
    entradas = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return aplicar_entradas(parametros_padrao(), entradas)


def salvar_resultados(resultado: Dict[str, Any], pasta: str = "resultados") -> None:
    destino = Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    if "resultados" in resultado:
        resultado["resultados"].to_csv(destino / "despacho_30min.csv", index=False)
    resumo = {k: v for k, v in resultado.items() if k != "resultados"}
    (destino / "resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    sistema = parametros_padrao()
    salvar_entradas(sistema, "entradas_exemplo.json")
    resultado = resolver(sistema)
    salvar_resultados(resultado)
    print({k: v for k, v in resultado.items() if k != "resultados"})
