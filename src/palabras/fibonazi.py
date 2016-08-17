'''
Created on 20/07/2016

@author: ernesto

https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=3895
'''
import logging
import array
from bitarray import bitarray

logger_cagada = None
# nivel_log = logging.ERROR
nivel_log = logging.DEBUG

def fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones):
    tamano_patron_referencia = 0
    tamano_patron_encontrar = 0
    matches_completos = {}
    
    tamano_patron_referencia = len(patron_referencia)
    tamano_patron_encontrar = len(patron_encontrar)

    logger_cagada.debug("patron ref %s patron enc %s" % (patron_referencia, patron_encontrar))
    
    assert(tamano_patron_referencia > tamano_patron_encontrar)

    
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
    assert(len(matches_completos) == 1)
    
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

if __name__ == '__main__':
    palabras_patron = []
    secuencia_grande=[]
    secuencia_no_grande=[]
    patron_referencia = bitarray("1011010110110")
    patron_encontrar = bitarray("110101101")
#    patron_referencia = bitarray("1011010110110")
#    patron_encontrar = bitarray("0110")
    patron_referencia.reverse()
    patron_encontrar.reverse()
    posiciones = {}
    
    
    logging.basicConfig(level=nivel_log)
    logger_cagada = logging.getLogger("asa")
    logger_cagada.setLevel(nivel_log)

    fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones)
    logger_cagada.debug("luna llena mi alma %s" % posiciones)

    fibonazi_genera_palabras_patron(palabras_patron)

#    logger_cagada.debug("homi %s"%palabras_patron)

    fibonazi_genera_sequencia_repeticiones(secuencia_grande, True)
#    logger_cagada.debug("la seq grande %s"%secuencia_grande)
    fibonazi_genera_sequencia_repeticiones(secuencia_no_grande, False)
#    logger_cagada.debug("la seq no grande %s"%secuencia_no_grande)
