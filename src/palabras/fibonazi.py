'''
Created on 20/07/2016

@author: ernesto

https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=3895
'''
import logging
import array
import operator
from bitarray import bitarray

logger_cagada = None
# nivel_log = logging.ERROR
nivel_log = logging.DEBUG

def caca_ordena_dick_llave(dick):
    return sorted(dick.items(), key=lambda cosa: cosa[0])

def caca_ordena_dick_valor(dick):
    return sorted(dick.items(), key=lambda cosa: operator.itemgetter(cosa[1], cosa[0]))

def fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones, matches_completos, corto_circuito=False):
    tamano_patron_referencia = 0
    tamano_patron_encontrar = 0
    
    tamano_patron_referencia = len(patron_referencia)
    tamano_patron_encontrar = len(patron_encontrar)

    logger_cagada.debug("patron ref %s patron enc %s" % (bitarray(list(reversed(patron_referencia))), bitarray(list(reversed(patron_encontrar)))))
    
#    assert(tamano_patron_referencia >= tamano_patron_encontrar)
    
    for pos_pat_ref in range(tamano_patron_referencia):
        posiciones_a_borrar = array.array("I")

        for pos_pat_ref_inicio, offset_valido in posiciones.items():
            pos_pat_ref_act = 0
            pos_pat_enc = 0
            
            pos_pat_ref_act = pos_pat_ref_inicio + offset_valido
            pos_pat_enc = offset_valido
            
            if(offset_valido == tamano_patron_encontrar):
                logger_cagada.debug("que calor que calor ya se encontro el patron completo empezando en %u" % (pos_pat_ref_inicio))
                matches_completos[pos_pat_ref_inicio] = True
                continue

            logger_cagada.debug("el patron que inicia en %u siwe vivo %u(%u) contra %u(%u)" % (pos_pat_ref_inicio, patron_referencia[pos_pat_ref_act], pos_pat_ref_act, patron_encontrar[pos_pat_enc], pos_pat_enc))
            
            if(patron_referencia[pos_pat_ref_act] == patron_encontrar[pos_pat_enc]):
                posiciones[pos_pat_ref_inicio] += 1
                if(posiciones[pos_pat_ref_inicio] == tamano_patron_encontrar):
                    logger_cagada.debug("knee deep ya no se buscara mas patron q inicia en %u" % (pos_pat_ref_inicio))
                    matches_completos[pos_pat_ref_inicio] = True
                    if(corto_circuito):
                        logger_cagada.debug("corto circuito activado asi q se sale")
                        break
                logger_cagada.debug("la posicion %u si la izo, avanzo a %u" % (pos_pat_ref_inicio, posiciones[pos_pat_ref_inicio]))
            else:
                logger_cagada.debug("la posicion %u no la izo" % pos_pat_ref_inicio)
                posiciones_a_borrar.append(pos_pat_ref_inicio)
            
        logger_cagada.debug("celso pina %s" % posiciones)
        for pos_a_bor in posiciones_a_borrar:
            logger_cagada.debug("baila con el rebelde %u" % pos_a_bor)
            del posiciones[pos_a_bor]
        
        if(patron_referencia[pos_pat_ref] == patron_encontrar[0]):
            posiciones[pos_pat_ref] = 1
            logger_cagada.debug("se inicia cagada %u(%u) vs %u(%u)" % (patron_referencia[pos_pat_ref], pos_pat_ref, patron_encontrar[0], 0))
    
    logger_cagada.debug("las posiciones finales son %s" % posiciones)
    logger_cagada.debug("los matches completos son %s" % matches_completos)
    assert(len(matches_completos) == 1 or len(matches_completos) == 0)
    
def fibonazi_genera_palabras_patron(palabras):
    tamano_palabra_actual = 0
    tamano_palabra_anterior_1 = 1
    tamano_palabra_anterior_2 = 1

    palabras.append(bitarray([False]))
    palabras.append(bitarray([True]))

    while tamano_palabra_actual < 100000:
        tamano_palabra_actual = tamano_palabra_anterior_1 + tamano_palabra_anterior_2
        palabras.append(palabras[-1] + palabras[-2])
#        logger_cagada.debug("el tamano actual de patron %s"%tamano_palabra_actual)
        assert(tamano_palabra_actual == len(palabras[-1]))
        tamano_palabra_anterior_2 = tamano_palabra_anterior_1
        tamano_palabra_anterior_1 = tamano_palabra_actual

    for palabra in palabras:
        palabra.reverse()
    
#    logger_cagada.debug("las palabras patron %s"%palabras)
    logger_cagada.debug("el tamano final %s" % tamano_palabra_actual)

def fibonazi_genera_sequencia_repeticiones(secuencia, generar_grande):
    secuencia.append(1)
    if(generar_grande):
        secuencia.append(2)
    else:
        secuencia.append(1)
    
    for idx_seq in range(3, 101):
        num_actual = 0
        
        num_actual = secuencia[-1] + secuencia[-2]
        if(idx_seq % 2):
            num_actual += 1
        secuencia.append(num_actual)

def fibonazi_encuentra_primera_aparicion_patron(patron_referencia, patrones_base):
    siguiente_coincidencia_doble = False
    tam_patron = 0
    idx_patron_tamano_coincide = 0
    tam_posiciones_match_completo = 0
    tam_componente_1 = 0
    idx_patron_encontrado = -1
    patron_tamano_coincide = []
    posiciones_match_completo_llave = []
    posiciones_patron = {}
    posiciones_match_completo = {}
    patron_base_1 = None
    patron_base_2 = None
    
    tam_patron = len(patron_referencia)
    
    for idx_patron, patron_base in enumerate(patrones_base):
        logger_cagada.debug("comaprando patron based tam %u con %u" % (len(patron_base), tam_patron))
        if(len(patron_base) >= tam_patron):
            idx_patron_tamano_coincide = idx_patron
            patron_tamano_coincide = patron_base
            break
        
    assert(len(patron_tamano_coincide) > 0)
    assert(len(patron_tamano_coincide) >= len(patron_referencia))
    
    patron_base_1 = patrones_base[idx_patron_tamano_coincide + 1]
    patron_base_2 = patrones_base[idx_patron_tamano_coincide + 2]
    
    
    fibonazi_compara_patrones(patron_tamano_coincide, patron_referencia , posiciones_patron, posiciones_match_completo)
    
    tam_posiciones_match_completo = len(posiciones_match_completo)
    
    assert(not tam_posiciones_match_completo or tam_posiciones_match_completo == 1)
    
    posiciones_match_completo_llave = caca_ordena_dick_llave(posiciones_match_completo)
    
    logger_cagada.debug("posiciones originales %s" % posiciones_patron)
    logger_cagada.debug("matches completos %s" % posiciones_match_completo)
    
    if(tam_posiciones_match_completo):
        idx_patron_encontrado = idx_patron_tamano_coincide 
        
        tam_componente_1 = len(patrones_base[idx_patron_tamano_coincide - 1])
        
        logger_cagada.debug("patron enc en base 0 %u" % idx_patron_encontrado)
        if(tam_patron <= tam_componente_1 and posiciones_match_completo_llave[0][0] >= 2):
            siguiente_coincidencia_doble = True
            logger_cagada.debug("ven bailalo la siwiente ocurrencia es d 2")
        
    else:
        posiciones_patron.clear()
        posiciones_match_completo.clear()
        
        fibonazi_compara_patrones(patron_base_1, patron_referencia , posiciones_patron, posiciones_match_completo)
        
        tam_posiciones_match_completo = len(posiciones_match_completo)
        
        assert(not tam_posiciones_match_completo or tam_posiciones_match_completo == 1)
        
        posiciones_match_completo_llave = caca_ordena_dick_llave(posiciones_match_completo)
        
        logger_cagada.debug("posiciones originales base 1 %s" % posiciones_patron)
        logger_cagada.debug("matches completos base 1 %s" % posiciones_match_completo_llave)
        
        if(tam_posiciones_match_completo):
            idx_patron_encontrado = idx_patron_tamano_coincide + 1
            logger_cagada.debug("patron enc en base 1 %u" % idx_patron_encontrado)
            
            tam_componente_1 = len(patrones_base[idx_patron_tamano_coincide + 1 - 1])
            if(tam_patron <= tam_componente_1 and posiciones_match_completo_llave[0][0] >= 2):
                siguiente_coincidencia_doble = True
                logger_cagada.debug("ven bailalo la siwiente ocurrencia es d 2")
        
        else:
            posiciones_patron.clear()
            posiciones_match_completo.clear()
            
            fibonazi_compara_patrones(patron_base_2, patron_referencia, posiciones_patron, posiciones_match_completo)
            
            tam_posiciones_match_completo = len(posiciones_match_completo)
            
            assert(tam_posiciones_match_completo == 1)
            
            posiciones_match_completo_llave = caca_ordena_dick_llave(posiciones_match_completo)
            
            logger_cagada.debug("posiciones originales base 2 %s" % posiciones_patron)
            logger_cagada.debug("matches completos base 2 %s" % posiciones_match_completo)
            
            tam_componente_1 = len(patrones_base[idx_patron_tamano_coincide + 2 - 1])
            if(tam_patron <= tam_componente_1 and posiciones_match_completo_llave[0][0] >= 2):
                siguiente_coincidencia_doble = True
                logger_cagada.debug("ven bailalo la siwiente ocurrencia es d 2")
            
            
            idx_patron_encontrado = idx_patron_tamano_coincide + 2
            
            
            logger_cagada.debug("patron enc en base 2 %u" % idx_patron_encontrado)
            
    assert(idx_patron_encontrado >= 0)
    
    logger_cagada.debug("ella no suelta idx %u siwiente doble %s" % (idx_patron_encontrado, siguiente_coincidencia_doble))
    
    return (idx_patron_encontrado, siguiente_coincidencia_doble)
        
def fibonazi_main(patron_referencia, patrones_base, idx_patrones_base_donde_buscar, repeticiones_inicio_lento, repeticiones_inicio_rapido):
    segunda_aparicion_doble = False
    idx_primera_aparicion_patron = 0
    separacion_primera_aparicion_y_donde_buscar = 0
    num_repeticiones = 0
    
    (idx_primera_aparicion_patron, segunda_aparicion_doble) = fibonazi_encuentra_primera_aparicion_patron(patron_referencia, patrones_base)
    
    separacion_primera_aparicion_y_donde_buscar = idx_patrones_base_donde_buscar - idx_primera_aparicion_patron
    
    logger_cagada.debug("la primera aparicion en %u, se busca en %u, diferencia %u" % (idx_primera_aparicion_patron, idx_patrones_base_donde_buscar, separacion_primera_aparicion_y_donde_buscar))
    
    if(not segunda_aparicion_doble):
        logger_cagada.debug("buscando en inicio lento")
        num_repeticiones = repeticiones_inicio_lento[separacion_primera_aparicion_y_donde_buscar]
    else:
        logger_cagada.debug("buscando en inicio lento")
        num_repeticiones = repeticiones_inicio_rapido[separacion_primera_aparicion_y_donde_buscar]
    
    logger_cagada.debug("")
    
    

if __name__ == '__main__':
    palabras_patron = []
    secuencia_grande = []
    secuencia_no_grande = []
    patron_encontrar = bitarray("110101101")
    patron_encontrar = bitarray("1101101011")
    patron_encontrar = bitarray("0101101011")
#    patron_encontrar = bitarray("0110")
    patron_encontrar.reverse()
    posiciones = {}
    
    
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=nivel_log, format=FORMAT)
    logger_cagada = logging.getLogger("asa")
    logger_cagada.setLevel(nivel_log)

#    fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones)
    logger_cagada.debug("luna llena mi alma %s" % posiciones)

    fibonazi_genera_palabras_patron(palabras_patron)

#    logger_cagada.debug("homi %s"%palabras_patron)

    fibonazi_genera_sequencia_repeticiones(secuencia_grande, True)
#    logger_cagada.debug("la seq grande %s"%secuencia_grande)
    fibonazi_genera_sequencia_repeticiones(secuencia_no_grande, False)
#    logger_cagada.debug("la seq no grande %s"%secuencia_no_grande)

fibonazi_encuentra_primera_aparicion_patron(patron_encontrar, palabras_patron)
