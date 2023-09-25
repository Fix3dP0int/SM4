#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <sys/time.h>
#include <pthread.h>
#define REC_FLS 1
#define REC_OFP 2
#define REC_IFP 4
#define REC_ENC 8
#define REC_DEC 16
#define REC_KEY 32
#define ENC 1
#define DEC 0
#define ROL(a, x) ((a) << (x) | (a) >> (32 - (x)))
typedef struct {
	uint32_t *plaintext;
	uint32_t *end;
	char mode;
	uint32_t *ciphertext;
} args_t;
uint32_t LEN = 1 << 28;
static uint32_t S_boxes_0[256];
static uint32_t S_boxes_1[256];
static uint32_t S_boxes_2[256];
static uint32_t S_boxes_3[256];
__m256i mmrk[32];
static char S_boxes_inited = 0;
void encrypt(const uint32_t X[32], uint32_t Y[32]);
void decrypt(const uint32_t X[32], uint32_t Y[32]);
void SM4(const uint32_t mk[4]);
static void S_boxes_init();
const uint32_t FK[4] = {
	0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc,
};
const uint32_t CK[32] = {
	0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
	0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
	0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
	0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
	0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
	0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
	0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
	0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279,
};
const uint8_t S_box[256] = {
	0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
	0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
	0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
	0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
	0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
	0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
	0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
	0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
	0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
	0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
	0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
	0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
	0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
	0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
	0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
	0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
};

static void S_boxes_init() {
    for (int i = 0; i < 256; i++) {
        uint32_t b = S_box[i] ^ S_box[i] << 2 ^ S_box[i] << 10 ^ S_box[i] << 18 ^ S_box[i] << 24;
        S_boxes_0[i] = b;
        S_boxes_1[i] = ROL(b, 010);
        S_boxes_2[i] = ROL(b, 020);
        S_boxes_3[i] = ROL(b, 030);
    }
    S_boxes_inited = 1;
}
void SM4(const uint32_t mk[4]) {
    if (!S_boxes_inited)
        S_boxes_init();
    uint32_t K[36] = {mk[0] ^ FK[0], mk[1] ^ FK[1], mk[2] ^ FK[2], mk[3] ^ FK[3]};
    for (int i = 0; i < 32; i++) {
        uint32_t a = K[i + 1] ^ K[i + 2] ^ K[i + 3] ^ CK[i];
        uint32_t b = S_box[a & 0xff] | S_box[a >> 010 & 0xff] << 010 | S_box[a >> 020 & 0xff] << 020 | S_box[a >> 030] << 030;
        uint32_t k = K[i + 4] = K[i] ^ b ^ ROL(b, 13) ^ ROL(b, 23);
        mmrk[i] = _mm256_set_epi32(k, k, k, k, k, k, k, k);
    }
}
void encrypt(const uint32_t X[32], uint32_t Y[32]){
    const __m256i mm0xff = _mm256_set_epi32(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
    __m256i mmT[4] = {
        _mm256_set_epi32(X[0x00], X[0x04], X[0x08], X[0x0c], X[0x10], X[0x14], X[0x18], X[0x1c]),
        _mm256_set_epi32(X[0x01], X[0x05], X[0x09], X[0x0d], X[0x11], X[0x15], X[0x19], X[0x1d]),
        _mm256_set_epi32(X[0x02], X[0x06], X[0x0a], X[0x0e], X[0x12], X[0x16], X[0x1a], X[0x1e]),
        _mm256_set_epi32(X[0x03], X[0x07], X[0x0b], X[0x0f], X[0x13], X[0x17], X[0x1b], X[0x1f]),
    };
    for (int i = 0; i < 32;) {
        __m256i mma;
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[1], mmT[2]), _mm256_xor_si256(mmT[3], mmrk[i++]));
        mmT[0] = _mm256_xor_si256(mmT[0], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[2], mmT[3]), _mm256_xor_si256(mmT[0], mmrk[i++]));
        mmT[1] = _mm256_xor_si256(mmT[1], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[3], mmT[0]), _mm256_xor_si256(mmT[1], mmrk[i++]));
        mmT[2] = _mm256_xor_si256(mmT[2], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[0], mmT[1]), _mm256_xor_si256(mmT[2], mmrk[i++]));
        mmT[3] = _mm256_xor_si256(mmT[3], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
    }
    uint32_t T_0[8], T_1[8], T_2[8], T_3[8];
    _mm256_storeu_si256((__m256i *)T_0, mmT[0]);
    _mm256_storeu_si256((__m256i *)T_1, mmT[1]);
    _mm256_storeu_si256((__m256i *)T_2, mmT[2]);
    _mm256_storeu_si256((__m256i *)T_3, mmT[3]);
    Y[0x00] = T_3[7];
    Y[0x04] = T_3[6];
    Y[0x08] = T_3[5];
    Y[0x0c] = T_3[4];
    Y[0x10] = T_3[3];
    Y[0x14] = T_3[2];
    Y[0x18] = T_3[1];
    Y[0x1c] = T_3[0];
    Y[0x01] = T_2[7];
    Y[0x05] = T_2[6];
    Y[0x09] = T_2[5];
    Y[0x0d] = T_2[4];
    Y[0x11] = T_2[3];
    Y[0x15] = T_2[2];
    Y[0x19] = T_2[1];
    Y[0x1d] = T_2[0];
    Y[0x02] = T_1[7];
    Y[0x06] = T_1[6];
    Y[0x0a] = T_1[5];
    Y[0x0e] = T_1[4];
    Y[0x12] = T_1[3];
    Y[0x16] = T_1[2];
    Y[0x1a] = T_1[1];
    Y[0x1e] = T_1[0];
    Y[0x03] = T_0[7];
    Y[0x07] = T_0[6];
    Y[0x0b] = T_0[5];
    Y[0x0f] = T_0[4];
    Y[0x13] = T_0[3];
    Y[0x17] = T_0[2];
    Y[0x1b] = T_0[1];
    Y[0x1f] = T_0[0];
}
void decrypt(const uint32_t X[32], uint32_t Y[32]){
    const __m256i mm0xff = _mm256_set_epi32(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff);
    __m256i mmT[4] = {
        _mm256_set_epi32(X[0x00], X[0x04], X[0x08], X[0x0c], X[0x10], X[0x14], X[0x18], X[0x1c]),
        _mm256_set_epi32(X[0x01], X[0x05], X[0x09], X[0x0d], X[0x11], X[0x15], X[0x19], X[0x1d]),
        _mm256_set_epi32(X[0x02], X[0x06], X[0x0a], X[0x0e], X[0x12], X[0x16], X[0x1a], X[0x1e]),
        _mm256_set_epi32(X[0x03], X[0x07], X[0x0b], X[0x0f], X[0x13], X[0x17], X[0x1b], X[0x1f]),
    };
    for (int i = 31; i >= 0;) {
        __m256i mma;
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[1], mmT[2]), _mm256_xor_si256(mmT[3], mmrk[i--]));
        mmT[0] = _mm256_xor_si256(mmT[0], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[2], mmT[3]), _mm256_xor_si256(mmT[0], mmrk[i--]));
        mmT[1] = _mm256_xor_si256(mmT[1], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[3], mmT[0]), _mm256_xor_si256(mmT[1], mmrk[i--]));
        mmT[2] = _mm256_xor_si256(mmT[2], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
        mma = _mm256_xor_si256(_mm256_xor_si256(mmT[0], mmT[1]), _mm256_xor_si256(mmT[2], mmrk[i--]));
        mmT[3] = _mm256_xor_si256(mmT[3], _mm256_xor_si256(
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_0, _mm256_and_si256(mma, mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_1, _mm256_and_si256(_mm256_srli_epi32(mma, 010), mm0xff), 4)
            ),
            _mm256_xor_si256(
                _mm256_i32gather_epi32(S_boxes_2, _mm256_and_si256(_mm256_srli_epi32(mma, 020), mm0xff), 4),
                _mm256_i32gather_epi32(S_boxes_3, _mm256_srli_epi32(mma, 030), 4)
            )
        ));
    }
    uint32_t T_0[8], T_1[8], T_2[8], T_3[8];
    _mm256_storeu_si256((__m256i *)T_0, mmT[0]);
    _mm256_storeu_si256((__m256i *)T_1, mmT[1]);
    _mm256_storeu_si256((__m256i *)T_2, mmT[2]);
    _mm256_storeu_si256((__m256i *)T_3, mmT[3]);
    Y[0x00] = T_3[7];
    Y[0x04] = T_3[6];
    Y[0x08] = T_3[5];
    Y[0x0c] = T_3[4];
    Y[0x10] = T_3[3];
    Y[0x14] = T_3[2];
    Y[0x18] = T_3[1];
    Y[0x1c] = T_3[0];
    Y[0x01] = T_2[7];
    Y[0x05] = T_2[6];
    Y[0x09] = T_2[5];
    Y[0x0d] = T_2[4];
    Y[0x11] = T_2[3];
    Y[0x15] = T_2[2];
    Y[0x19] = T_2[1];
    Y[0x1d] = T_2[0];
    Y[0x02] = T_1[7];
    Y[0x06] = T_1[6];
    Y[0x0a] = T_1[5];
    Y[0x0e] = T_1[4];
    Y[0x12] = T_1[3];
    Y[0x16] = T_1[2];
    Y[0x1a] = T_1[1];
    Y[0x1e] = T_1[0];
    Y[0x03] = T_0[7];
    Y[0x07] = T_0[6];
    Y[0x0b] = T_0[5];
    Y[0x0f] = T_0[4];
    Y[0x13] = T_0[3];
    Y[0x17] = T_0[2];
    Y[0x1b] = T_0[1];
    Y[0x1f] = T_0[0];
}
void *threading(void *pbuf) {
	uint32_t *plaintext = ((args_t *)pbuf)->plaintext, *ciphertext = ((args_t *)pbuf)->ciphertext, *end = ((args_t *)pbuf)->end;
	char mode = ((args_t *)pbuf)->mode;
	while(plaintext < end){
		if (mode == 1) encrypt(plaintext, ciphertext);
		else decrypt(ciphertext, plaintext);
		plaintext += 32;
		ciphertext += 32;
	}
	return NULL;
}
void multithreading(uint32_t *data, uint32_t *ciphertext, size_t len, int subthreads, char mode){
	pthread_t ptid[subthreads];
	args_t pbuf[subthreads];
	size_t part = (len + subthreads) / (subthreads + 1);
	uint32_t *plaintext = data, *end = data + len;
	for (int i = 0; i < subthreads; i++){
		pbuf[i].plaintext = plaintext;
		pbuf[i].ciphertext = ciphertext;
		pbuf[i].mode = mode;
		plaintext += part;
		pbuf[i].end = plaintext;
		ciphertext += part;
		pthread_create(ptid + i, NULL, threading, pbuf + i);
	}
	while(plaintext < end){
		if (mode == 1) encrypt(plaintext, ciphertext);
		else decrypt(ciphertext, plaintext);
		plaintext += 32;
		ciphertext += 32;
	}
	for (int i = 0; i < subthreads; i++) {
		pthread_join(ptid[i], NULL);
	}
}
int main(int argc, char *argv[]) {
	int rec = REC_KEY;
	FILE *ifp, *ofp;
	uint32_t key[4] = {0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210};
	for (int i = 1; (rec & REC_FLS) == 0 && i < argc; i++)
		if (argv[i][0] == '-')
			if (argv[i][1] == 'k' && argv[i][2] == '\0')
				if ((rec & REC_KEY) == 0 && i + 1 < argc)
					rec |= REC_KEY;
				else
					rec |= REC_FLS;
			else if (argv[i][1] == 'd' && argv[i][2] == '\0')
				if ((rec & (REC_DEC | REC_ENC)) == 0)
					rec |= REC_DEC;
				else
					rec |= REC_FLS;
			else if (argv[i][1] == 'e' && argv[i][2] == '\0')
				if ((rec & (REC_DEC | REC_ENC)) == 0)
					rec |= REC_ENC;
				else
					rec |= REC_FLS;
			else if (argv[i][1] == 'i' && argv[i][2] == '\0')
				if ((rec & REC_IFP) == 0 && i + 1 < argc && (ifp = fopen(argv[++i], "rb")))
					rec |= REC_IFP;
				else
					rec |= REC_FLS;
			else if (argv[i][1] == 'o' && argv[i][2] == '\0')
				if ((rec & REC_OFP) == 0 && i + 1 < argc && (ofp = fopen(argv[++i], "wb")))
					rec |= REC_OFP;
				else
					rec |= REC_FLS;
			else
				rec |= REC_FLS;
		else
			rec |= REC_FLS;
	if ((rec & (REC_DEC | REC_ENC)) == 0 || (rec & REC_KEY) == 0 || (rec & REC_FLS) != 0) {
		fprintf(stderr, "usage: %s [-i IFILENAME] [-o OFILENAME] (-e | -d) <-k KEY>\n", argv[0]);
		if ((rec & REC_IFP) != 0)
			fclose(ifp);
		if ((rec & REC_OFP) != 0)
			fclose(ofp);
		return 1;
	}
	if ((rec & REC_IFP) == 0)
		ifp = stdin;
	if ((rec & REC_OFP) == 0)
		ofp = stdout;
	if ((rec & REC_DEC) != 0) {
		// uint32_t rk[32], data[4], res[4], next[4];
		// key_expand(key, rk);
		// fread(data, 4, 4, ifp);
		// while (fread(next, 4, 4, ifp)) {
		// 	SM4_decrypt(rk, data, res);
		// 	fwrite(res, 4, 4, ofp);
		// 	data[0] = next[0];
		// 	data[1] = next[1];
		// 	data[2] = next[2];
		// 	data[3] = next[3];
		// }
		// SM4_decrypt(rk, data, res);
		// fwrite(res, 1, 16 - ((uint8_t *)res)[15], ofp);
	}
	else {
		// size_t len;
		// uint32_t rk[32], data[4], res[4];
		// key_expand(key, rk);
		// struct timeval start, end;
		// double t = 0;
		// while ((len = fread(data, 1, 16, ifp)) == 16) {
		// 	//SM4_encrypt(rk, data, res);
		// 	//fwrite(res, 4, 4, ofp);
		// 	gettimeofday(&start, NULL);
		// 	SM4_encrypt(rk, data, res);
		// 	gettimeofday(&end, NULL);
		// 	fwrite(res, 4, 4, ofp);
		// 	t += end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1e-6; 
		// }
		// printf("time:%fs", t);
		// ((uint8_t *)data)[15] = 16 - len;
		// SM4_encrypt(rk, data, res);
		// //fwrite(res, 4, 4, ofp);
        uint32_t *data = calloc(LEN , sizeof(uint32_t));// 1 GB
		uint32_t *res = calloc(LEN , sizeof(uint32_t));
		// uint32_t *temp = calloc(LEN , sizeof(uint32_t));
		SM4(key);
		fread(data, 1, LEN, ifp);
        for (int i = 0; i < 16; i++) {
			struct timeval start, end;
			gettimeofday(&start, NULL);
			multithreading(data, res, LEN, i, ENC);
			gettimeofday(&end, NULL);
			printf("%d threads:\n", i + 1);
			printf("time = %f s\n", end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) * 1e-6);
		}
	}

	fclose(ifp);
	fclose(ofp);
}