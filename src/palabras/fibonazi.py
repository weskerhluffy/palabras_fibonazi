'''
Created on 20/07/2016

@author: ernesto

https://uva.onlinejudge.org/index.php?option=onlinejudge&page=show_problem&problem=3895
'''
import logging
import array
import operator
import fileinput
import argparse
import sys


logger_cagada = None
nivel_log = logging.ERROR
nivel_log = logging.DEBUG

__version__ = "3.1.5"

__author__ = "Scott Griffiths"

import numbers
import copy
import re
import binascii
import mmap
import os
import struct
import collections

byteorder = sys.byteorder

bytealigned = False

MAX_CHARS = 250

CACHE_SIZE = 1000

class Error(Exception):

    def __init__(self, *params):
        self.msg = params[0] if params else ''
        self.params = params[1:]

    def __str__(self):
        if self.params:
            return self.msg.format(*self.params)
        return self.msg


class ReadError(Error, IndexError):

    def __init__(self, *params):
        Error.__init__(self, *params)


class InterpretError(Error, ValueError):

    def __init__(self, *params):
        Error.__init__(self, *params)


class ByteAlignError(Error):

    def __init__(self, *params):
        Error.__init__(self, *params)


class CreationError(Error, ValueError):

    def __init__(self, *params):
        Error.__init__(self, *params)


class ConstByteStore(object):

    __slots__ = ('offset', '_rawarray', 'bitlength')

    def __init__(self, data, bitlength=None, offset=None):
        self._rawarray = data
        if offset is None:
            offset = 0
        if bitlength is None:
            bitlength = 64 * len(data) - offset
        self.offset = offset
        logger_cagada.debug("El offset al crear %s" % self.offset)
        self.bitlength = bitlength
        logger_cagada.debug("El bitlen al crear %s" % self.bitlength)

#    @profile
    def getbit(self, pos):
        assert 0 <= pos < self.bitlength, "la posi %d supera el limite %d" % (pos, self.bitlength)
        logger_cagada.debug("el patron es %s" % self._rawarray)
        logger_cagada.debug("q vergas es offset %s" % (self.offset))
        logger_cagada.debug("la pos deseada es %s" % (pos))
        byte, bit = divmod(pos, 64)
        return bool(self._rawarray[byte] & (1 << bit))

    def getbyte(self, pos):
        return self._rawarray[pos]

    def getbyteslice(self, start, end):
        c = self._rawarray[start:end]
        return c

    @property
    def bytelength(self):
        if not self.bitlength:
            return 0
        eb = (self.bitlength - 1) // 64
        return eb + 1

    def __copy__(self):
        return ByteStore(self._rawarray[:], self.bitlength, self.offset)

    @property
    def byteoffset(self):
        return self.offset // 8

    @property
    def rawbytes(self):
        return self._rawarray


class ByteStore(ConstByteStore):
    pass


def offsetcopy(s, newoffset):
    assert 0 <= newoffset < 64
    if not s.bitlength:
        return copy.copy(s)
    else:
        newdata = array.array("Q")
        d = s._rawarray
#        if newoffset <= restante_a_la_der:
        shiftleft = newoffset
        bits_libres_a_la_izq = 64 - s.bitlength % 64
        if(bits_libres_a_la_izq == 64):
            bits_libres_a_la_izq = 0
        logger_cagada.debug("los bitchs libres a la izq %u" % (bits_libres_a_la_izq))
        logger_cagada.debug("el byte original inicial %s con bytelen %u" % (bin(s.getbyte(0)), s.bytelength))
        newdata.append(s.getbyte(0) << (shiftleft) & 0xffffffffffffffff)
        logger_cagada.debug("se añadio al inicio %s" % (bin(newdata[-1])))
        for x in range(1, s.bytelength):
            newdata.append(((d[x] << shiftleft) & 0xffffffffffffffff) + \
                           (d[x - 1] >> (64 - shiftleft)))
            logger_cagada.debug("añadiendo byte %s de byte orig %u" % (bin(newdata[-1]), x))
        if shiftleft > bits_libres_a_la_izq:
            offset_ultimo_byte = ((s.bitlength % 64) - (shiftleft - bits_libres_a_la_izq))
            if(offset_ultimo_byte < 0):
                offset_ultimo_byte = 64 - shiftleft
            newdata.append(d[-1] >> offset_ultimo_byte & 0xffffffffffffffff)
            logger_cagada.debug("se añadio al final (por salir una cabecilla) %s" % (bin(newdata[-1])))
        
#        else:  # newoffset > s._offset % 8

#        logger_cagada.debug("nuevo offs %u y rest a la izq %u"%(newoffset, s.bitlength % 8))
#        shiftright = newoffset
#        newdata.append(s.getbyte(0) << (8-shiftright))
#        logger_cagada.debug("l 1er byte %s q c recorre %u, resultado %s" % (bin(s.getbyte(0)), shiftright, newdata[-1]))
#        for x in range(0, len(d)-1):
#            logger_cagada.debug("la parte de enfrente %s la de atras %s"%(d[x + 1] << (8 - shiftright),d[x] >> shiftright))
#            newdata.append(((d[x + 1] << (8 - shiftright)) & 0xff) + \
#                           (d[x] >> shiftright))
#        newdata.append(d[-1] >> shiftright)

#        bits_in_last_byte = newoffset - restante_a_la_der if restante_a_la_der < newoffset else 0
#        if bits_in_last_byte:
#            newdata.append((d[s.bytelength - 1] << (8 - shiftright)) & 0xff)
        new_s = ByteStore(newdata, s.bitlength, 0)
        return new_s


def equal(a, b):
    a_bitlength = a.bitlength
    b_bitlength = b.bitlength
    if a_bitlength != b_bitlength:
        return False
    if not a_bitlength:
        assert b_bitlength == 0
        return True
    if (a.offset % 8) > (b.offset % 8):
        a, b = b, a
    a_bitoff = a.offset % 8
    b_bitoff = b.offset % 8
    a_byteoffset = a.byteoffset
    b_byteoffset = b.byteoffset
    a_bytelength = a.bytelength
    b_bytelength = b.bytelength
    da = a._rawarray
    db = b._rawarray

    if da is db and a.offset == b.offset:
        return True

    if a_bitoff == b_bitoff:
        bits_spare_in_last_byte = 8 - (a_bitoff + a_bitlength) % 8
        if bits_spare_in_last_byte == 8:
            bits_spare_in_last_byte = 0
        if a_bytelength == 1:
            a_val = ((da[a_byteoffset] << a_bitoff) & 0xff) >> (8 - a_bitlength)
            b_val = ((db[b_byteoffset] << b_bitoff) & 0xff) >> (8 - b_bitlength)
            return a_val == b_val
        if da[a_byteoffset] & (0xff >> a_bitoff) != db[b_byteoffset] & (0xff >> b_bitoff):
            return False
        b_a_offset = b_byteoffset - a_byteoffset
        for x in range(1 + a_byteoffset, a_byteoffset + a_bytelength - 1):
            if da[x] != db[b_a_offset + x]:
                return False
        return (da[a_byteoffset + a_bytelength - 1] >> bits_spare_in_last_byte == 
                db[b_byteoffset + b_bytelength - 1] >> bits_spare_in_last_byte)

    assert a_bitoff != b_bitoff
    shift = b_bitoff - a_bitoff
    if b_bytelength == 1:
        assert a_bytelength == 1
        a_val = ((da[a_byteoffset] << a_bitoff) & 0xff) >> (8 - a_bitlength)
        b_val = ((db[b_byteoffset] << b_bitoff) & 0xff) >> (8 - b_bitlength)
        return a_val == b_val
    if a_bytelength == 1:
        assert b_bytelength == 2
        a_val = ((da[a_byteoffset] << a_bitoff) & 0xff) >> (8 - a_bitlength)
        b_val = ((db[b_byteoffset] << 8) + db[b_byteoffset + 1]) << b_bitoff
        b_val &= 0xffff
        b_val >>= 16 - b_bitlength
        return a_val == b_val

    if (da[a_byteoffset] & (0xff >> a_bitoff)) >> shift != db[b_byteoffset] & (0xff >> b_bitoff):
        return False
    for x in range(1, b_bytelength - 1):
        b_val = db[b_byteoffset + x]
        a_val = ((da[a_byteoffset + x - 1] << 8) + da[a_byteoffset + x]) >> shift
        a_val &= 0xff
        if a_val != b_val:
            return False

    final_b_bits = (b.offset + b_bitlength) % 8
    if not final_b_bits:
        final_b_bits = 8
    b_val = db[b_byteoffset + b_bytelength - 1] >> (8 - final_b_bits)
    final_a_bits = (a.offset + a_bitlength) % 8
    if not final_a_bits:
        final_a_bits = 8
    if b.bytelength > a_bytelength:
        assert b_bytelength == a_bytelength + 1
        a_val = da[a_byteoffset + a_bytelength - 1] >> (8 - final_a_bits)
        a_val &= 0xff >> (8 - final_b_bits)
        return a_val == b_val
    assert a_bytelength == b_bytelength
    a_val = da[a_byteoffset + a_bytelength - 2] << 8
    a_val += da[a_byteoffset + a_bytelength - 1]
    a_val >>= (8 - final_a_bits)
    a_val &= 0xff >> (8 - final_b_bits)
    return a_val == b_val


class MmapByteArray(object):
    pass

BYTE_REVERSAL_DICT = dict()

try:
    xrange
    for i in range(256):
        BYTE_REVERSAL_DICT[i] = chr(int("{0:08b}".format(i)[::-1], 2))
except NameError:
    for i in range(256):
        BYTE_REVERSAL_DICT[i] = bytes([int("{0:08b}".format(i)[::-1], 2)])
    from io import IOBase as file
    xrange = range
    basestring = str

LEADING_OCT_CHARS = len(oct(1)) - 1

def tidy_input_string(s):
    s = ''.join(s.split()).lower()
    return s

INIT_NAMES = ('uint', 'int', 'ue', 'se', 'sie', 'uie', 'hex', 'oct', 'bin', 'bits',
              'uintbe', 'intbe', 'uintle', 'intle', 'uintne', 'intne',
              'float', 'floatbe', 'floatle', 'floatne', 'bytes', 'bool', 'pad')

TOKEN_RE = re.compile(r'(?P<name>' + '|'.join(INIT_NAMES) + 
                      r')((:(?P<len>[^=]+)))?(=(?P<value>.*))?$', re.IGNORECASE)
DEFAULT_UINT = re.compile(r'(?P<len>[^=]+)?(=(?P<value>.*))?$', re.IGNORECASE)

MULTIPLICATIVE_RE = re.compile(r'(?P<factor>.*)\*(?P<token>.+)')

LITERAL_RE = re.compile(r'(?P<name>0(x|o|b))(?P<value>.+)', re.IGNORECASE)

STRUCT_PACK_RE = re.compile(r'(?P<endian><|>|@)?(?P<fmt>(?:\d*[bBhHlLqQfd])+)$')

STRUCT_SPLIT_RE = re.compile(r'\d*[bBhHlLqQfd]')

REPLACEMENTS_BE = {'b': 'intbe:8', 'B': 'uintbe:8',
                   'h': 'intbe:16', 'H': 'uintbe:16',
                   'l': 'intbe:32', 'L': 'uintbe:32',
                   'q': 'intbe:64', 'Q': 'uintbe:64',
                   'f': 'floatbe:32', 'd': 'floatbe:64'}
REPLACEMENTS_LE = {'b': 'intle:8', 'B': 'uintle:8',
                   'h': 'intle:16', 'H': 'uintle:16',
                   'l': 'intle:32', 'L': 'uintle:32',
                   'q': 'intle:64', 'Q': 'uintle:64',
                   'f': 'floatle:32', 'd': 'floatle:64'}

PACK_CODE_SIZE = {'b': 1, 'B': 1, 'h': 2, 'H': 2, 'l': 4, 'L': 4,
                  'q': 8, 'Q': 8, 'f': 4, 'd': 8}

_tokenname_to_initialiser = {'hex': 'hex', '0x': 'hex', '0X': 'hex', 'oct': 'oct',
                             '0o': 'oct', '0O': 'oct', 'bin': 'bin', '0b': 'bin',
                             '0B': 'bin', 'bits': 'auto', 'bytes': 'bytes', 'pad': 'pad'}

OCT_TO_BITS = ['{0:03b}'.format(i) for i in xrange(8)]

BIT_COUNT = dict(zip(xrange(256), [bin(i).count('1') for i in xrange(256)]))


class Bits(object):

    __slots__ = ('_datastore')

    def __init__(self, auto=None, length=None, offset=None, **kwargs):
        pass

    def __new__(cls, auto=None, length=None, offset=None, _cache={}, **kwargs):
        try:
            if isinstance(auto, basestring):
                try:
                    return _cache[auto]
                except KeyError:
                    x = object.__new__(Bits)
                    try:
                        _, tokens = (None, None)
                    except ValueError as e:
                        raise CreationError(*e.args)
                    x._datastore = ConstByteStore(bytearray(0), 0, 0)
                    for token in tokens:
                        x._datastore._appendstore(Bits._init_with_token(*token)._datastore)
                    assert x._assertsanity()
                    if len(_cache) < CACHE_SIZE:
                        _cache[auto] = x
                    return x
            if type(auto) == Bits:
                return auto
        except TypeError:
            pass
        x = super(Bits, cls).__new__(cls)
        x._initialise(auto, length, offset, **kwargs)
        return x

    def _initialise(self, auto, length, offset, **kwargs):
        if length is not None and length < 0:
            raise CreationError("bitstring length cannot be negative.")
        if offset is not None and offset < 0:
            raise CreationError("offset must be >= 0.")
        if auto is not None:
            self._initialise_from_auto(auto, length, offset)
            return
        if not kwargs:
            if length is not None and length != 0:
                num_words = (length + 63) // 64
                data = array.array("Q") 
                self._setbytes_unsafe(data, length, 0)
                return
            self._setbytes_unsafe(array.array("Q"), 0, 0)
            return
        k, v = kwargs.popitem()
        try:
            init_without_length_or_offset[k](self, v)
            if length is not None or offset is not None:
                raise CreationError("Cannot use length or offset with this initialiser.")
        except KeyError:
            try:
                init_with_length_only[k](self, v, length)
                if offset is not None:
                    raise CreationError("Cannot use offset with this initialiser.")
            except KeyError:
                if offset is None:
                    offset = 0
                try:
                    init_with_length_and_offset[k](self, v, length, offset)
                except KeyError:
                    raise CreationError("Unrecognised keyword '{0}' used to initialise.", k)

    def _initialise_from_auto(self, auto, length, offset):
        if offset is None:
            offset = 0
        self._setauto(auto, length, offset)
        return

    def __copy__(self):
        return self
    
    def __add__(self, bs):
        parte_izq = self._copy()
        parte_der = bs._copy()
        
        parte_izq_ds = parte_izq._datastore
        parte_der_ds = parte_der._datastore
        
        tam_orig = parte_izq_ds.bitlength 
        if not tam_orig:
            return
        logger_cagada.debug("el tam orig de bit %u" % tam_orig)
        logger_cagada.debug("appendeando pumpkin hea %s" % (",".join(bin(x) + " (" + hex(x) + ")" for x in parte_izq_ds._rawarray)))
        logger_cagada.debug("we wont pre offset %u bitchlen %u" % (parte_izq_ds.offset, parte_izq_ds.bitlength))
        
        if(parte_der_ds.bitlength % 64):
            store1 = offsetcopy(parte_izq_ds, parte_der_ds.bitlength % 64)
            logger_cagada.debug("your drees %s se recorrio %u" % (",".join(bin(x) + " (" + hex(x) + ")" for x in store1._rawarray), parte_der_ds.bitlength % 64))
            logger_cagada.debug("uniendo ultimo bit de der %s con primer bit de izq %s" % (bin(parte_der_ds._rawarray[-1]), bin(store1._rawarray[0])))
            parte_der_ds._rawarray[-1] |= store1._rawarray[0]
    #            joinval = (self._rawarray.pop() & (255 ^ (255 >> store.offset)) | 
    #                       (store.getbyte(0) & (255 >> store.offset)))
            parte_der_ds._rawarray.extend(store1._rawarray[1:])
        else:
            parte_der_ds._rawarray.extend(parte_izq_ds._rawarray)
        
        parte_der_ds.bitlength += tam_orig
#        parte_der_ds.bytelength = (parte_der_ds.bitlength + 7) // 8
        
        logger_cagada.debug("el nuevo bitch len %u se le sumo %u, el original era %u" % (parte_der_ds.bitlength, tam_orig, parte_der_ds.bitlength - tam_orig))
        logger_cagada.debug("la cadenita keda finalmente %s de bytelen %u" % (parte_der.bin, parte_der_ds.bytelength))
        return parte_der

    def __getitem__(self, key):
        return self._datastore.getbit(key)

    def __len__(self):
        return self._getlength()

    def __str__(self):
        length = self.len
        if not length:
            return ''
        if length > MAX_CHARS * 4:
            return ''.join(('0x', self._readhex(MAX_CHARS * 4, 0), '...'))
        if length < 32 and length % 4 != 0:
            return '0b' + self.bin
        if not length % 4:
            return '0x' + self.hex
        bits_at_end = length % 4
        return ''.join(('0x', self._readhex(length - bits_at_end, 0),
                        ', ', '0b',
                        self._readbin(bits_at_end, length - bits_at_end)))

    def __eq__(self, bs):
        try:
            bs = Bits(bs)
        except TypeError:
            return False
        return equal(self._datastore, bs._datastore)


    def _setauto(self, s, length, offset):
        if isinstance(s, Bits):
            if length is None:
                length = s.len - offset
            self._setbytes_unsafe(s._datastore.rawbytes, length, s._offset + offset)
            return
        if isinstance(s, file):
            if offset is None:
                offset = 0
            if length is None:
                length = os.path.getsize(s.name) * 8 - offset
            byteoffset, offset = divmod(offset, 8)
            bytelength = (length + byteoffset * 8 + offset + 7) // 8 - byteoffset
            m = MmapByteArray(s, bytelength, byteoffset)
            if length + byteoffset * 8 + offset > m.filelength * 8:
                raise CreationError("File is not long enough for specified "
                                    "length and offset.")
            self._datastore = ConstByteStore(m, length, offset)
            return
        if length is not None:
            raise CreationError("The length keyword isn't applicable to this initialiser.")
        if offset:
            raise CreationError("The offset keyword isn't applicable to this initialiser.")
        if isinstance(s, basestring):
            bs = self._converttobitstring(s)
            assert bs._offset == 0
            self._setbytes_unsafe(bs._datastore.rawbytes, bs.length, 0)
            return
        if isinstance(s, (bytes, bytearray)):
            self._setbytes_unsafe(bytearray(s), len(s) * 8, 0)
            return
        if isinstance(s, array.array):
            b = s.tostring()
            self._setbytes_unsafe(bytearray(b), len(b) * 8, 0)
            return
        if isinstance(s, numbers.Integral):
            if s < 0:
                msg = "Can't create bitstring of negative length {0}."
                raise CreationError(msg, s)
            data = bytearray((s + 7) // 8)
            self._datastore = ByteStore(data, s, 0)
            return
        if isinstance(s, collections.Iterable):
            self._setbin_unsafe(''.join(str(int(bool(x))) for x in s))
            return
        raise TypeError("Cannot initialise bitstring from {0}.".format(type(s)))

    def _assertsanity(self):
        assert self.len >= 0
        assert (self.len + 63) // 64 == self._datastore.bytelength, "el calculo %u (len %u) contra lo q es %u" % ((self.len + 63) // 64, self.len , self._datastore.bytelength)
        return True

    def _setbytes_safe(self, data, length=None, offset=0):
        data = bytearray(data)
        if length is None:
            length = len(data) * 8 - offset
            self._datastore = ByteStore(data, length, offset)
        else:
            if length + offset > len(data) * 8:
                msg = "Not enough data present. Need {0} bits, have {1}."
                raise CreationError(msg, length + offset, len(data) * 8)
            if length == 0:
                self._datastore = ByteStore(bytearray(0))
            else:
                self._datastore = ByteStore(data, length, offset)

    def _setbytes_unsafe(self, data, length, offset):
        logger_cagada.debug("la del moño %s con offset %d" % (data, offset))
        self._datastore = ByteStore(data[:], length, offset)
        assert self._assertsanity()
        
    
        
    def _getbytes(self):
        if self.len % 8:
            raise InterpretError("Cannot interpret as bytes unambiguously - "
                                 "not multiple of 8 bits.")
        return self._readbytes(self.len, 0)


    def _setbin_safe(self, binstring):
        logger_cagada.debug("convirtiendo la cadena %s a bytearray? " % binstring)
        binstring = tidy_input_string(binstring)
        binstring = binstring.replace('0b', '')
        self._setbin_unsafe(binstring)

    def _setbin_unsafe(self, binstring):
        length = len(binstring)
        boundary = ((length + 63) // 64) * 64
        padded_binstring = '0' * (boundary - length) + binstring \
                           if len(binstring) < boundary else binstring
        logger_cagada.debug("evil all of the t %s" % padded_binstring)
        tam_padded_str = len(padded_binstring)
        logger_cagada.debug("primer cadena para el bitch %s " % padded_binstring[8:16])
        assert not tam_padded_str % 64
        try:
            bytelist = [int(padded_binstring[x - 64 :x], 2)
                        for x in xrange(len(padded_binstring) , 0, -64)]
        except ValueError:
            raise CreationError("Invalid character in bin initialiser {0}.", binstring)
        logger_cagada.debug("la lista de bytes %s" % bytelist)
        word_array = array.array("Q", bytelist)
        self._setbytes_unsafe(word_array, length, 0)

    def _readbin(self, length, start):
        if not length:
            return ''
        startbyte = 0
        endbyte = self._datastore.bytelength - 1
        b = bytearray(reversed(bytearray(self._datastore.getbyteslice(startbyte, endbyte + 1))))
        logger_cagada.debug("la cadena revertida %s" % b)
        try:
            c = binascii.hexlify(b)
            logger_cagada.debug("if i cut %s"%c)
            c = "{:0{}b}".format(int(binascii.hexlify(b), 16), 8 * len(b))
        except TypeError:
            c = "{0:0{1}b}".format(int(binascii.hexlify(str(b)), 16), 8 * len(b))
#        return c[startoffset:startoffset + length]
        return c

    def _getbin(self):
        return self._readbin(self.len, 0)

    def _readhex(self, length, start):
        if length % 4:
            raise InterpretError("Cannot convert to hex unambiguously - "
                                           "not multiple of 4 bits.")
        if not length:
            return ''
        s = self._slice(start, start + length).tobytes()
        try:
            s = s.hex()  # Available in Python 3.5
        except AttributeError:
            s = str(binascii.hexlify(s).decode('utf-8'))
        return s[:-1] if (length // 4) % 2 else s

    def _gethex(self):
        """Return the hexadecimal representation as a string prefixed with '0x'.

        Raises an InterpretError if the bitstring's length is not a multiple of 4.

        """
        return self._readhex(self.len, 0)

    def _getoffset(self):
        return self._datastore.offset

    def _getlength(self):
        return self._datastore.bitlength

    def _copy(self):
        s_copy = self.__class__()
        s_copy._setbytes_unsafe(self._datastore.getbyteslice(0, self._datastore.bytelength),
                                self.len, self._offset)
        return s_copy

    def _sethex(self, hexstring):
        """Reset the bitstring to have the value given in hexstring."""
        hexstring = tidy_input_string(hexstring)
        # remove any 0x if present
        hexstring = hexstring.replace('0x', '')
        length = len(hexstring)
        if length % 2:
            hexstring += '0'
        try:
            try:
                data = bytearray.fromhex(hexstring)
            except TypeError:
                # Python 2.6 needs a unicode string (a bug). 2.7 and 3.x work fine.
                data = bytearray.fromhex(unicode(hexstring))
        except ValueError:
            raise CreationError("Invalid symbol in hex initialiser.")
        self._setbytes_unsafe(data, length * 4, 0)

    def _slice(self, start, end):
        if end == start:
            return self.__class__()
        offset = self._offset
        startbyte, newoffset = divmod(start + offset, 8)
        endbyte = (end + offset - 1) // 8
        bs = self.__class__()
        bs._setbytes_unsafe(self._datastore.getbyteslice(startbyte, endbyte + 1), end - start, newoffset)
        return bs

    def _append(self, bs):
        self._datastore._appendstore(bs._datastore)

    def _prepend(self, bs):
        """Prepend a bitstring to the current bitstring."""
        self._datastore._prependstore(bs._datastore)

    def _reverse(self):
        n = [BYTE_REVERSAL_DICT[b] for b in self._datastore.rawbytes]
        n.reverse()
        newoffset = 8 - (self._offset + self.len) % 8
        if newoffset == 8:
            newoffset = 0
        self._setbytes_unsafe(bytearray().join(n), self.length, newoffset)

    def _validate_slice(self, start, end):
        if start is None:
            start = 0
        elif start < 0:
            start += self.len
        if end is None:
            end = self.len
        elif end < 0:
            end += self.len
        if not 0 <= end <= self.len:
            raise ValueError("end is not a valid position in the bitstring.")
        if not 0 <= start <= self.len:
            raise ValueError("start is not a valid position in the bitstring.")
        if end < start:
            raise ValueError("end must not be less than start.")
        return start, end

    def tobytes(self):
        d = offsetcopy(self._datastore, 0).rawbytes
        unusedbits = 8 - self.len % 8
        if unusedbits != 8:
            d[-1] &= (0xff << unusedbits)
        return bytes(d)


    _offset = property(_getoffset)

    len = property(_getlength,
                   doc="""The length of the bitstring in bits. Read only.
                      """)
    length = property(_getlength,
                      doc="""The length of the bitstring in bits. Read only.
                      """)
    bin = property(_getbin,
                   doc="""The bitstring as a binary string. Read only.
                   """)


name_to_read = {
                'hex': Bits._readhex,
                'bin': Bits._readbin,
                }

init_with_length_and_offset = {
                               }

init_with_length_only = {
                         }

init_without_length_or_offset = {
                                 'bin': Bits._setbin_safe,
                                 }


class BitArray(Bits):

    __slots__ = ()

    __hash__ = None

    def __init__(self, auto=None, length=None, offset=None, **kwargs):
        if not isinstance(self._datastore, ByteStore):
            self._ensureinmemory()

    def __new__(cls, auto=None, length=None, offset=None, **kwargs):
        x = super(BitArray, cls).__new__(cls)
        y = Bits.__new__(BitArray, auto, length, offset, **kwargs)
        x._datastore = y._datastore
        return x

    def reverse(self, start=None, end=None):
        start, end = self._validate_slice(start, end)
        if start == 0 and end == self.len:
            self._reverse()
            return
        s = self._slice(start, end)
        s._reverse()
        self[start:end] = s

    bytes = property(Bits._getbytes, Bits._setbytes_safe,
                     doc="""The bitstring as a ordinary string. Read and write.
                      """)
    hex = property(Bits._gethex, Bits._sethex,
                   doc="""The bitstring as a hexadecimal string. Read and write.
                       """)



class ConstBitStream(Bits):

    __slots__ = ('_pos')

    def __init__(self, auto=None, length=None, offset=None, **kwargs):
        self._pos = 0

    def __new__(cls, auto=None, length=None, offset=None, **kwargs):
        x = super(ConstBitStream, cls).__new__(cls)
        x._initialise(auto, length, offset, **kwargs)
        return x
    

class BitStream(ConstBitStream, BitArray):

    __slots__ = ()

    __hash__ = None

    def __init__(self, auto=None, length=None, offset=None, **kwargs):
        self._pos = 0
        if not isinstance(self._datastore, ByteStore):
            self._ensureinmemory()

    def __new__(cls, auto=None, length=None, offset=None, **kwargs):
        x = super(BitStream, cls).__new__(cls)
        x._initialise(auto, length, offset, **kwargs)
        return x
    

ConstBitArray = Bits
BitString = BitStream

__all__ = ['ConstBitArray', 'ConstBitStream', 'BitStream', 'BitArray',
           'Bits', 'BitString', 'pack', 'Error', 'ReadError',
           'InterpretError', 'ByteAlignError', 'CreationError', 'bytealigned']








def caca_ordena_dick_llave(dick):
    return sorted(dick.items(), key=lambda cosa: cosa[0])

def caca_ordena_dick_valor(dick):
    return sorted(dick.items(), key=lambda cosa: operator.itemgetter(cosa[1], cosa[0]))

# @profile
def fibonazi_compara_patrones(patron_referencia, patron_encontrar, posiciones, matches_completos, corto_circuito=True, pegate=0):
    tamano_patron_referencia = 0
    tamano_patron_encontrar = 0
    posicion_coincidente = 0
    patron_referencia_offset = 0
    patron_referencia_raw = []
    patron_encontrar_raw = []
    posiciones_tmp = {}
    
    patron_referencia_raw = patron_referencia._datastore._rawarray
    patron_encontrar_raw = patron_encontrar._datastore._rawarray
    
    tamano_patron_referencia = len(patron_referencia)
    tamano_patron_encontrar = patron_encontrar._datastore.bytelength
    
    logger_cagada.debug("tam patron ref (bits) %u tam patron enc (bytes) %u" % (tamano_patron_referencia, tamano_patron_encontrar))

    logger_cagada.debug("patron ref %s patron enc %s" % (patron_referencia.bin, patron_encontrar.bin))
    
    primer_byte_patron_enc = patron_encontrar_raw[0]
    
    sobrante_patron_enc = len(patron_encontrar) % 64
    logger_cagada.debug("sobrante de bitches %u del total %u" % (sobrante_patron_enc, len(patron_encontrar)))
    maskara_ultimo_byte_patron_enc = 0xffffffffffffffff >> (64 - sobrante_patron_enc) if sobrante_patron_enc else 0xffffffffffffffff
    ultimo_idx_patron_enc = tamano_patron_encontrar - 1
    lomote_iteracion = tamano_patron_referencia - sobrante_patron_enc
    if(sobrante_patron_enc):
        logger_cagada.debug("hay sobrante")
        lomote_iteracion += 1
    else:
        lomote_iteracion = tamano_patron_referencia - 63
    
    logger_cagada.debug("iterndo  asta %u" % (lomote_iteracion))
    for pos_pat_ref in range(0, lomote_iteracion):

        byte = pos_pat_ref >> 6
        bit = pos_pat_ref & 63
        
        logger_cagada.debug("el byte %u el bit %u de %u" % (byte, bit, pos_pat_ref))
        byte_actual_patron_ref = (patron_referencia_raw[byte] >> bit) 
        if(byte < len(patron_referencia_raw) - 1):
            logger_cagada.debug("es q cual %s de %s" % (bin((patron_referencia_raw[byte + 1] << (64 - bit)) & 0xffffffffffffffff), bin(patron_referencia_raw[byte + 1])))
            byte_actual_patron_ref |= ((patron_referencia_raw[byte + 1] << (64 - bit)) & 0xffffffffffffffff)

        logger_cagada.debug("i aora solo tengo %s con byte %u y bit %u de %s" % (bin(byte_actual_patron_ref), byte, bit, bin(patron_referencia_raw[byte])))
        
        for pos_pat_ref_inicio, offset_valido in posiciones_tmp.setdefault(pos_pat_ref, []):
            
            assert offset_valido < tamano_patron_encontrar, "el offset valido %u y el tam encontra %u" % (offset_valido, tamano_patron_encontrar)
            
            maskara_comparacion = maskara_ultimo_byte_patron_enc
            if(offset_valido < ultimo_idx_patron_enc):
                maskara_comparacion = 0xffffffffffffffff

            logger_cagada.debug("el patron que inicia en %u siwe vivo %s(%u) contra %s(%u)" % (pos_pat_ref_inicio, bin(byte_actual_patron_ref & maskara_comparacion), pos_pat_ref, bin(patron_encontrar_raw[offset_valido]), offset_valido))
            
            logger_cagada.debug("en este cuadro %s" % bin(maskara_comparacion))
            if((byte_actual_patron_ref & maskara_comparacion) == patron_encontrar_raw[offset_valido]):
                if(offset_valido + 1 == tamano_patron_encontrar):
                    logger_cagada.debug("knee deep ya no se buscara mas patron q inicia en %u, tamanio en bytes %u " % (pos_pat_ref_inicio, tamano_patron_encontrar))
                    matches_completos[ pos_pat_ref_inicio ] = True
                    if(corto_circuito):
                        logger_cagada.debug("corto circuito activado asi q se sale")
                        break
                else:
                    assert (offset_valido + 1) < tamano_patron_encontrar, "como es posible, el offset es %u, cuando el tam del patron enc es %u" % (offset_valido + 1, tamano_patron_encontrar)
                    posiciones_tmp.setdefault(pos_pat_ref + 64, []).append((pos_pat_ref_inicio, offset_valido + 1))
                    logger_cagada.debug("la posicion %u si la izo, avanzo a %u" % (pos_pat_ref_inicio, posiciones_tmp[pos_pat_ref + 64][-1][1]))
            else:
                logger_cagada.debug("la posicion %u no la izo" % pos_pat_ref_inicio)
                pass
            
        if(corto_circuito and matches_completos):
            break
        logger_cagada.debug("celso pina %s" % posiciones_tmp)
        
        maskara_comparacion = maskara_ultimo_byte_patron_enc
        if(tamano_patron_encontrar > 1):
            maskara_comparacion = 0xffffffffffffffff
        logger_cagada.debug("polka pelazon %s" % bin(maskara_comparacion))
        if((byte_actual_patron_ref & maskara_comparacion) == primer_byte_patron_enc):
            if(tamano_patron_encontrar == 1):
                matches_completos[pos_pat_ref ] = True
                if(corto_circuito):
                    logger_cagada.debug("corto circuito activado asi q se sale con el 1er bit")
                    break
            else:
                posiciones_tmp.setdefault(pos_pat_ref + 64, []).append((pos_pat_ref, 1))
                logger_cagada.debug("se inicia cagada %s(%u) vs %s(%u)" % (bin(byte_actual_patron_ref), pos_pat_ref , bin(primer_byte_patron_enc), 0))
 

    if(nivel_log == logging.DEBUG):
        logger_cagada.debug("las posiciones tmp %s" % posiciones_tmp)
        if(posiciones_tmp):
            for _, saltos in posiciones_tmp.items():
                for inicio_match, tam_match in saltos:
                    if(inicio_match not in posiciones):
                        posiciones[inicio_match] = tam_match
                    if(tam_match > posiciones[inicio_match]):
                        posiciones[inicio_match] = tam_match

    for posicion in matches_completos.keys():
        posiciones[posicion] = tamano_patron_encontrar

    logger_cagada.debug("las posiciones finales son %s" % posiciones)
    logger_cagada.debug("los matches completos son %s" % matches_completos)

    if(not pegate):
        assert len(matches_completos) == 1 or len(matches_completos) == 0, "los matches son %s, los patrones %s y %s" % (matches_completos, patron_referencia.bin, patron_referencia.bin)
    else:
        assert len(matches_completos) == pegate, "los matches son %s, lo esperado %u los patrones %s y %s" % (matches_completos, pegate, patron_referencia.bin, patron_encontrar.bin)
    
def fibonazi_genera_palabras_patron(palabras, tam_palabra_a_idx_patron):
    tamano_palabra_actual = 0
    tamano_palabra_anterior_1 = 1
    tamano_palabra_anterior_2 = 1

    palabras.append(BitArray([False]))
    palabras.append(BitArray([True]))

    tam_palabra_a_idx_patron.append(0)
    tam_palabra_a_idx_patron.append(0)

    while tamano_palabra_actual < 300000:
        tamano_palabra_actual = tamano_palabra_anterior_1 + tamano_palabra_anterior_2
        palabras.append(palabras[-1] + palabras[-2])

        for _ in range(tamano_palabra_anterior_1 + 1, tamano_palabra_actual + 1):
            tam_palabra_a_idx_patron.append(len(palabras) - 1)
        assert(tamano_palabra_actual == len(palabras[-1]))
        tamano_palabra_anterior_2 = tamano_palabra_anterior_1
        tamano_palabra_anterior_1 = tamano_palabra_actual

    logger_cagada.debug("el tamano final %s" % tamano_palabra_actual)
    logger_cagada.debug("los primeros patrones%s" % ([x.bin for x in palabras[0:14]]))
    logger_cagada.debug("los mapeos de tamanos de palabras %s" % (tam_palabra_a_idx_patron[0:20]))

def fibonazi_genera_sequencia_repeticiones(secuencia, generar_grande):
    secuencia.append(1)
    if(generar_grande < 2):
        secuencia.append(1)
    else:
        secuencia.append(2)
        
    
    for idx_seq in range(3, 102):
        num_actual = 0
        
        num_actual = secuencia[-1] + secuencia[-2]
        if(generar_grande and (idx_seq % 2 or generar_grande == 2)):
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

    logger_cagada.debug("io t kiero dar am %u" % tam_patron)
    
    if(tam_patron == 1):
        if(patron_referencia[0] == False):
            return (0, False)
        else:
            return (1, False)

    if(tam_patron == 2):
        logger_cagada.debug("conparando %s con ref %s" % (patron_referencia, BitArray([True, False]).bin))
        if(patron_referencia[0] == False and patron_referencia[1] == True):
            return (2, False)
    
    idx_patron_tamano_coincide = tam_palabra_a_idx_patron[tam_patron]
    logger_cagada.debug("el idx q l corresponde por tam %u" % idx_patron_tamano_coincide)
    patron_tamano_coincide = patrones_base[idx_patron_tamano_coincide]

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
        if(posiciones_match_completo_llave[0][0] >= 2):
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
            if(posiciones_match_completo_llave[0][0] >= 2):
                siguiente_coincidencia_doble = True
                logger_cagada.debug("ven bailalo la siwiente ocurrencia es d 2")
        
        else:
            posiciones_patron.clear()
            posiciones_match_completo.clear()
            
            fibonazi_compara_patrones(patron_base_2, patron_referencia, posiciones_patron, posiciones_match_completo)
            
            tam_posiciones_match_completo = len(posiciones_match_completo)
            
            if(not tam_posiciones_match_completo):
                return(-1, False)
            
            assert(tam_posiciones_match_completo == 1)
            
            posiciones_match_completo_llave = caca_ordena_dick_llave(posiciones_match_completo)
            
            logger_cagada.debug("posiciones originales base 2 %s" % posiciones_patron)
            logger_cagada.debug("matches completos base 2 %s" % posiciones_match_completo)
            
            tam_componente_1 = len(patrones_base[idx_patron_tamano_coincide + 2 - 1])
            if(posiciones_match_completo_llave[0][0] >= 2):
                siguiente_coincidencia_doble = True
                logger_cagada.debug("ven bailalo la siwiente ocurrencia es d 2")
            
            
            idx_patron_encontrado = idx_patron_tamano_coincide + 2
            
            
            logger_cagada.debug("patron enc en base 2 %u" % idx_patron_encontrado)
            
    assert(idx_patron_encontrado >= 0)
    
    logger_cagada.debug("ella no suelta idx %u siwiente doble %s" % (idx_patron_encontrado, siguiente_coincidencia_doble))
    
    return (idx_patron_encontrado, siguiente_coincidencia_doble)
        
def fibonazi_main(patron_referencia, patrones_base, idx_patrones_base_donde_buscar, repeticiones_inicio_lento, repeticiones_inicio_rapido, repeticiones_inicio_muy_lento):
    segunda_aparicion_doble = False
    idx_primera_aparicion_patron = 0
    separacion_primera_aparicion_y_donde_buscar = 0
    num_repeticiones = 0
    
    (idx_primera_aparicion_patron, segunda_aparicion_doble) = fibonazi_encuentra_primera_aparicion_patron(patron_referencia, patrones_base)
    
    if(idx_primera_aparicion_patron == -1):
        return 0
    
    separacion_primera_aparicion_y_donde_buscar = idx_patrones_base_donde_buscar - idx_primera_aparicion_patron
    
    if(separacion_primera_aparicion_y_donde_buscar < 0):
        return 0

    logger_cagada.debug("la primera aparicion en %u, se busca en %u, diferencia %u" % (idx_primera_aparicion_patron, idx_patrones_base_donde_buscar, separacion_primera_aparicion_y_donde_buscar))

    assert(separacion_primera_aparicion_y_donde_buscar >= 0)
    
    if(not segunda_aparicion_doble):
        if(len (patron_referencia) == 2 and patron_referencia[0] == False and patron_referencia[1] == True):
            logger_cagada.debug("buscando en inicio muuuy lento pos %u" % separacion_primera_aparicion_y_donde_buscar)
            num_repeticiones = repeticiones_inicio_muy_lento[separacion_primera_aparicion_y_donde_buscar]
        else:
            logger_cagada.debug("buscando en inicio lento pos %u" % separacion_primera_aparicion_y_donde_buscar)
            num_repeticiones = repeticiones_inicio_lento[separacion_primera_aparicion_y_donde_buscar]
    else:
        logger_cagada.debug("buscando en inicio rapido %u" % separacion_primera_aparicion_y_donde_buscar)
        num_repeticiones = repeticiones_inicio_rapido[separacion_primera_aparicion_y_donde_buscar]
    
    logger_cagada.debug("el num de repeticiones de %s en la pos %u es %u" % (patron_referencia.bin, idx_patrones_base_donde_buscar, num_repeticiones))

    assert(num_repeticiones)

    if(nivel_log == logging.DEBUG and len(patron_referencia) != 1):
        posiciones_patron = {}
        posiciones_match_completo = {}
        if(segunda_aparicion_doble):
            pegate = 2
        else:
            pegate = 1
        logger_cagada.debug("validando q efectivamente el siwiente idx de donde aparece %u (%s) tenga %u repeticones" % (idx_primera_aparicion_patron + 1, patrones_base[idx_primera_aparicion_patron + 1].bin, pegate))
        fibonazi_compara_patrones(patrones_base[idx_primera_aparicion_patron + 1], patron_referencia, posiciones_patron, posiciones_match_completo, pegate=pegate, corto_circuito=False)
        assert((segunda_aparicion_doble and len(posiciones_match_completo) == 2) or (not segunda_aparicion_doble and len(posiciones_match_completo) == 1))
    
        if(idx_patrones_base_donde_buscar < 25):
            posiciones_patron = {}
            posiciones_match_completo = {}
            pegate = num_repeticiones
            fibonazi_compara_patrones(patrones_base[idx_patrones_base_donde_buscar], patron_referencia, posiciones_patron, posiciones_match_completo, pegate=pegate, corto_circuito=False)
            assert(len(posiciones_match_completo) == pegate)
    
    return num_repeticiones

def fibonazi_genere_todos_los_pedazos(palabrota, tam_ini=1, tam_fin=100000):
    ya_generadas = {}
    tam_pal = len(palabrota)
    for tam_act in range(tam_ini, tam_fin + 1):
        for pos_ini in range(tam_pal - tam_act):
            pala_act = palabrota[pos_ini:pos_ini + tam_act]
            butes = pala_act.tobytes()
            if(butes not in ya_generadas):
                idx_primera_aparicion = tam_palabra_a_idx_patron[tam_act]
                ya_generadas[butes] = True
                for idx_donde_buscar in range(idx_primera_aparicion, 101):
                    print("%u" % idx_donde_buscar)
                    print("%s" % (pala_act.bin))


if __name__ == '__main__':
    linea_idx = 0
    idx_a_buscar = 0
    palabras_patron = []
    secuencia_grande = []
    secuencia_no_grande = []
    secuencia_peke = []
    tam_palabra_a_idx_patron = []
    parser = None
    args = None
    

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=nivel_log, format=FORMAT)
    logger_cagada = logging.getLogger("asa")
    logger_cagada.setLevel(nivel_log)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nadena", help="i rompe tu camisa", action="store_true")

    args = parser.parse_args()
    
#    a = BitArray(bin="111011111111110")
#    a += BitArray(bin="10101000001")
   
#    b = BitArray(bin="11100101")
#    b += BitArray(bin="101")
   
#    logger_cagada.debug("la mierda %s" % b)
    
#    sys.exit()
    

    fibonazi_genera_palabras_patron(palabras_patron, tam_palabra_a_idx_patron)


    if(args.nadena):
        fibonazi_genere_todos_los_pedazos(palabras_patron[25], tam_ini=1, tam_fin=100)
        fibonazi_genere_todos_los_pedazos(palabras_patron[25], tam_ini=100000, tam_fin=100000)
        sys.exit()



    fibonazi_genera_sequencia_repeticiones(secuencia_grande, 2)
    logger_cagada.debug("la seq grande %s" % secuencia_grande)
    fibonazi_genera_sequencia_repeticiones(secuencia_no_grande, 1)
    logger_cagada.debug("la seq no grande %s" % secuencia_no_grande)
    fibonazi_genera_sequencia_repeticiones(secuencia_peke, 0)
    
    for linea in sys.stdin:
        if(not linea.strip()):
            continue
        if(not linea_idx % 2):
            idx_a_buscar = int(linea.strip())

        else:
            num_repeticiones = 0
            patron_encontrar = None
            
            logger_cagada.debug("si alguna vez %s no dig" % (linea.strip()))
            assert(linea.strip())
#            patron_encontrar = BitArray(bin="01")
            patron_encontrar = BitArray(bin=linea.strip())
            logger_cagada.debug("vinimos para liar %u %s" % (idx_a_buscar, patron_encontrar.bin))

            num_repeticiones = fibonazi_main(patron_encontrar, palabras_patron, idx_a_buscar, secuencia_no_grande, secuencia_grande, secuencia_peke)
            print("Case #%u %u" % (linea_idx / 2 + 1, num_repeticiones))
        linea_idx += 1
    

