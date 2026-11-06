0000000000000000 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus>:
   0:	mov    0x4(%rsi),%ecx
   3:	movq   (%rdx),%xmm0
   7:	movdqa 0x0(%rip),%xmm1        # f <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xf>
   f:	pand   %xmm0,%xmm1
  13:	pshufd $0x55,%xmm1,%xmm2
  18:	movd   %xmm2,%r8d
  1d:	movd   %xmm1,%r9d
  22:	sub    %ecx,%r9d
  25:	sub    %ecx,%r8d
  28:	movd   0x8(%rdx),%xmm4
  2d:	movd   %xmm4,%eax
  31:	and    $0x7fffffff,%eax
  36:	mov    %eax,%r10d
  39:	sub    %ecx,%r10d
  3c:	or     %r9d,%r10d
  3f:	or     %r8d,%r10d
  42:	cmp    0x8(%rsi),%r10d
  46:	ja     256 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x256>
  4c:	paddd  0x0(%rip),%xmm1        # 54 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x54>
  54:	movdqa 0x0(%rip),%xmm5        # 5c <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x5c>
  5c:	pand   %xmm1,%xmm5
  60:	paddd  0x0(%rip),%xmm5        # 68 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x68>
  68:	movaps 0x0(%rip),%xmm3        # 6f <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x6f>
  6f:	addps  %xmm5,%xmm3
  72:	addps  0x0(%rip),%xmm5        # 79 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x79>
  79:	movss  (%rsi),%xmm2
  7d:	divps  %xmm5,%xmm3
  80:	add    $0x4afb0d,%eax
  85:	mov    %eax,%ecx
  87:	shr    $0x17,%ecx
  8a:	and    $0x7fffff,%eax
  8f:	add    $0x3f3504f3,%eax
  94:	movd   %eax,%xmm6
  98:	movss  0x0(%rip),%xmm5        # a0 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xa0>
  a0:	addss  %xmm6,%xmm5
  a4:	addss  0x0(%rip),%xmm6        # ac <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xac>
  ac:	add    $0xffffff81,%ecx
  af:	divss  %xmm6,%xmm5
  b3:	movaps %xmm5,%xmm6
  b6:	mulss  %xmm5,%xmm6
  ba:	movss  0x0(%rip),%xmm7        # c2 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xc2>
  c2:	mulss  %xmm6,%xmm7
  c6:	addss  0x0(%rip),%xmm7        # ce <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xce>
  ce:	mulss  %xmm6,%xmm7
  d2:	addss  0x0(%rip),%xmm7        # da <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xda>
  da:	mulss  %xmm5,%xmm6
  de:	mulss  %xmm7,%xmm6
  e2:	xorps  %xmm7,%xmm7
  e5:	cvtsi2ss %ecx,%xmm7
  e9:	mulss  0x0(%rip),%xmm5        # f1 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0xf1>
  f1:	addss  %xmm7,%xmm5
  f5:	addss  %xmm6,%xmm5
  f9:	mulss  %xmm2,%xmm5
  fd:	movss  0x0(%rip),%xmm6        # 105 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x105>
 105:	addss  %xmm5,%xmm6
 109:	movd   %xmm6,%ecx
 10d:	add    $0x20,%ecx
 110:	mov    %ecx,%edx
 112:	addss  0x0(%rip),%xmm6        # 11a <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x11a>
 11a:	and    $0x3f,%edx
 11d:	mov    0x0(%rip),%rax        # 124 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x124>
 124:	movss  (%rax,%rdx,8),%xmm8
 12a:	subss  %xmm6,%xmm5
 12e:	movss  0x0(%rip),%xmm7        # 136 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x136>
 136:	mulss  %xmm5,%xmm7
 13a:	addss  0x0(%rip),%xmm7        # 142 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x142>
 142:	mulss  %xmm5,%xmm7
 146:	and    $0xffffffc0,%ecx
 149:	shl    $0x11,%ecx
 14c:	add    $0x3f800000,%ecx
 152:	movd   %ecx,%xmm5
 156:	mulss  %xmm8,%xmm7
 15b:	addss  %xmm8,%xmm7
 160:	mulss  %xmm5,%xmm7
 164:	movdqa 0x0(%rip),%xmm5        # 16c <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x16c>
 16c:	movdqa %xmm5,%xmm6
 170:	pandn  %xmm4,%xmm6
 174:	movaps %xmm3,%xmm4
 177:	mulps  %xmm3,%xmm4
 17a:	psrld  $0x17,%xmm1
 17f:	paddd  0x0(%rip),%xmm1        # 187 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x187>
 187:	movaps 0x0(%rip),%xmm8        # 18f <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x18f>
 18f:	mulps  %xmm4,%xmm8
 193:	addps  0x0(%rip),%xmm8        # 19b <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x19b>
 19b:	andps  %xmm5,%xmm7
 19e:	mulps  %xmm4,%xmm8
 1a2:	addps  0x0(%rip),%xmm8        # 1aa <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x1aa>
 1aa:	por    %xmm7,%xmm6
 1ae:	mulps  %xmm3,%xmm4
 1b1:	mulps  %xmm8,%xmm4
 1b5:	cvtdq2ps %xmm1,%xmm1
 1b8:	mulps  0x0(%rip),%xmm3        # 1bf <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x1bf>
 1bf:	addps  %xmm1,%xmm3
 1c2:	addps  %xmm4,%xmm3
 1c5:	shufps $0x0,%xmm2,%xmm2
 1c9:	mulps  %xmm3,%xmm2
 1cc:	movaps 0x0(%rip),%xmm3        # 1d3 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x1d3>
 1d3:	addps  %xmm2,%xmm3
 1d6:	movdqa 0x0(%rip),%xmm1        # 1de <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x1de>
 1de:	paddd  %xmm3,%xmm1
 1e2:	movd   %xmm1,%ecx
 1e6:	and    $0x3f,%ecx
 1e9:	pshufd $0x55,%xmm1,%xmm4
 1ee:	movd   %xmm4,%edx
 1f2:	addps  0x0(%rip),%xmm3        # 1f9 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x1f9>
 1f9:	and    $0x3f,%edx
 1fc:	subps  %xmm3,%xmm2
 1ff:	movaps 0x0(%rip),%xmm3        # 206 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x206>
 206:	mulps  %xmm2,%xmm3
 209:	addps  0x0(%rip),%xmm3        # 210 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x210>
 210:	mulps  %xmm2,%xmm3
 213:	pslld  $0x11,%xmm1
 218:	pand   0x0(%rip),%xmm1        # 220 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x220>
 220:	paddd  0x0(%rip),%xmm1        # 228 <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x228>
 228:	movss  (%rax,%rcx,8),%xmm2
 22d:	movss  (%rax,%rdx,8),%xmm4
 232:	unpcklps %xmm4,%xmm2
 235:	mulps  %xmm2,%xmm3
 238:	addps  %xmm2,%xmm3
 23b:	mulps  %xmm1,%xmm3
 23e:	andps  %xmm5,%xmm3
 241:	pandn  %xmm0,%xmm5
 245:	por    %xmm3,%xmm5
 249:	movq   %xmm5,(%rdi)
 24d:	movd   %xmm6,0x8(%rdi)
 252:	mov    %rdi,%rax
 255:	ret
 256:	push   %rbx
 257:	mov    %rdi,%rbx
 25a:	call   25f <<moxcms::trc::ToneCurveEvaluatorPureGamma as moxcms::trc::ToneCurveEvaluator>::evaluate_tristimulus+0x25f>
 25f:	mov    %rbx,%rax
 262:	pop    %rbx
 263:	ret

Disassembly of section .text._RNvXs9_NtCs8N3SupDdS21_6moxcms3trcNtB5_24ToneCurveEvaluatorLinearNtB5_18ToneCurveEvaluator14evaluate_value:
