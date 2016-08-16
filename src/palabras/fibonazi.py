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
    posiciones_a_borrar = array.array("I")
    tamano_patron_referencia = 0
    tamano_patron_encontrar = 0
    
    tamano_patron_referencia = len(patron_referencia)
    tamano_patron_encontrar = len(patron_encontrar)
    
    assert(tamano_patron_referencia > tamano_patron_encontrar)
    
    for pos_pat_ref in range(tamano_patron_referencia):
        for pos_pat_ref_inicio, offset_valido in posiciones.items():
            pos_pat_ref_act = 0
            pos_pat_enc = 0
            
            logger_cagada.debug("el patron que inicia en %u siwe vivo" % pos_pat_ref_inicio)
            
            pos_pat_ref_act = pos_pat_ref_inicio + offset_valido
            pos_pat_enc = offset_valido
            
            if(offset_valido == tamano_patron_encontrar):
                logger_cagada.debug("que calor que calor ya se encontro el patron completo empezando en %u" % (pos_pat_ref_inicio))
                continue
            
            if(patron_referencia[pos_pat_ref_act] == patron_encontrar[pos_pat_enc]):
                posiciones[pos_pat_ref_inicio] += 1
                logger_cagada.debug("la posicion %u si la izo, avanzo a %u" % (pos_pat_ref_inicio, posiciones[pos_pat_ref_inicio]))
            else:
                logger_cagada.debug("la posicion %u no la izo" % pos_pat_ref_inicio)
                posiciones_a_borrar.append(pos_pat_ref_inicio)
            
        for pos_a_bor in posiciones_a_borrar:
            del posiciones[pos_a_bor]
        
        if(patron_referencia[pos_pat_ref] == patron_encontrar[0]):
            posiciones[pos_pat_ref] = 1
            logger_cagada.debug("se inicia cagada en %u" % pos_pat_ref)
    
    

if __name__ == '__main__':
    patron_referencia = bitarray("10110")
    patron_encontrar = bitarray("101")
    posiciones = {}
    
    
    logging.basicConfig(level=nivel_log)
    logger_cagada = logging.getLogger("asa")
    logger_cagada.setLevel(nivel_log)

    fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones)
