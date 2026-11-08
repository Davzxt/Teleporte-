"""Texto → fonemas do português brasileiro (inventário do corpus FalaBrasil hts-br)."""
import re

ACENTO = {"á": "a", "à": "a", "â": "a", "ã": "a~", "é": "E", "ê": "e", "í": "i", "ó": "O", "ô": "o", "õ": "o~", "ú": "u"}
VOG = set("aeiouáàâãéêíóôõú")
FR = set("eiéêí")


def palavra(w):
    out, i, n = [], 0, len(w)
    while i < n:
        c = w[i]; nx = w[i + 1] if i + 1 < n else ""; n2 = w[i + 2] if i + 2 < n else ""
        pv = w[i - 1] if i else ""
        fim = (i + 1 == n) or (i + 2 == n and nx == "s")
        if w[i:i + 2] == "ão": out += ["a~", "w~"]; i += 2; continue
        if w[i:i + 2] == "õe": out += ["o~", "j~"]; i += 2; continue
        if w[i:i + 2] == "ãe": out += ["a~", "j~"]; i += 2; continue
        if w[i:i + 2] == "ch": out.append("S"); i += 2; continue
        if w[i:i + 2] == "lh": out.append("L"); i += 2; continue
        if w[i:i + 2] == "nh": out.append("J"); i += 2; continue
        if w[i:i + 2] == "rr": out.append("R"); i += 2; continue
        if w[i:i + 2] in ("ss", "sç"): out.append("s"); i += 2; continue
        if w[i:i + 2] == "qu":
            out += ["k"] if nx and n2 in FR else ["k", "w"]; i += 2; continue
        if w[i:i + 2] == "gu" and n2 in FR and n2: out.append("g"); i += 2; continue
        if c in VOG:
            v = ACENTO.get(c, c)
            # nasal: vogal + m/n + (consoante ou fim)
            if nx in ("m", "n") and (not n2 or n2 not in VOG) and nx != "":
                if w[i + 1:] in ("m", "ns", "m") and v in ("e", "E"):
                    out += ["e~", "j~"]; i += 2; continue
                if w[i + 1:] == "m" and v == "a":
                    out += ["a~", "w~"]; i += 2; continue
                out.append(v.rstrip("~") + "~" if v[0] in "aeiou" else v); i += 2; continue
            # semivogais em ditongos decrescentes
            if c in "iu" and pv in VOG and pv not in "iu" and not (nx in ("m", "n")):
                out.append("j" if c == "i" else "w"); i += 1; continue
            # átonas finais
            if fim and c == "e" and n > 2: v = "i"
            if fim and c == "o" and n > 2: v = "u"
            out.append(v); i += 1; continue
        if c in "td": out.append(c); i += 1; continue          # vira tS/dZ antes de [i] (pós-processo)
        if c == "c": out.append("s" if nx in FR and nx else "k"); i += 1; continue
        if c == "ç": out.append("s"); i += 1; continue
        if c == "g": out.append("Z" if nx in FR and nx else "g"); i += 1; continue
        if c == "j": out.append("Z"); i += 1; continue
        if c == "x": out.append("s" if (pv == "e" and nx and nx not in VOG) else "S"); i += 1; continue
        if c == "h": i += 1; continue
        if c == "s":
            out.append("z" if (pv in VOG and nx in VOG and pv and nx) else "s"); i += 1; continue
        if c == "r":
            if i == 0 or pv in "nls": out.append("R")
            elif not nx or nx not in VOG: out.append("X")
            else: out.append("r")
            i += 1; continue
        if c == "l":
            out.append("w" if (not nx or nx not in VOG) else "l"); i += 1; continue
        if c == "z": out.append("s" if fim else "z"); i += 1; continue
        if c in "pbfvkmn": out.append(c); i += 1; continue
        i += 1
    # -te/-de finais: "ti"/"di" → tS/dZ
    for k in range(len(out) - 1):
        if out[k] in ("t", "d") and out[k + 1] in ("i", "i~"):
            out[k] = "tS" if out[k] == "t" else "dZ"
    return out


def frase(texto):
    seq = ["#"]
    for tok in re.findall(r"[a-záàâãéêíóôõúç]+|[?.!,;]", texto.lower().replace("-", " ")):
        if tok in "?.!,;":
            if seq[-1] != "#":
                seq.append("#")
        else:
            seq += palavra(tok)
    if seq[-1] != "#":
        seq.append("#")
    return seq


if __name__ == "__main__":
    for t in ["Você ouviu isso?", "O vento não bate na porta.", "Base, aqui é o explorador. Cheguei ao planeta.",
              "Tem alguma coisa ali, em cima do morro.", "Eu vou até lá.", "O céu tem dois sóis, e o vento não para."]:
        print(t, "→", " ".join(frase(t)))
