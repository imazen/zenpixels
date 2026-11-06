0000000000000000 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb>:
   0:	push   %r15
   2:	push   %r14
   4:	push   %rbx
   5:	sub    $0x20,%rsp
   9:	movss  (%rdx),%xmm5
   d:	xorps  %xmm0,%xmm0
  10:	ucomiss %xmm0,%xmm5
  13:	jne    2f <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2f>
  15:	jp     2f <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2f>
  17:	movss  (%rsi),%xmm6
  1b:	ucomiss %xmm0,%xmm6
  1e:	jne    13d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x13d>
  24:	jp     13d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x13d>
  2a:	jmp    3b1 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3b1>
  2f:	movd   %xmm5,%ecx
  33:	and    $0x7fffffff,%ecx
  39:	mov    %ecx,%eax
  3b:	sub    0x4(%rsi),%eax
  3e:	movss  (%rsi),%xmm6
  42:	cmp    0x8(%rsi),%eax
  45:	ja     3b1 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3b1>
  4b:	add    $0x4afb0d,%ecx
  51:	mov    %ecx,%eax
  53:	and    $0x7fffff,%ecx
  59:	add    $0x3f3504f3,%ecx
  5f:	movd   %ecx,%xmm1
  63:	movss  0x0(%rip),%xmm0        # 6b <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x6b>
  6b:	addss  %xmm1,%xmm0
  6f:	addss  0x0(%rip),%xmm1        # 77 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x77>
  77:	divss  %xmm1,%xmm0
  7b:	movaps %xmm0,%xmm1
  7e:	mulss  %xmm0,%xmm1
  82:	movss  0x0(%rip),%xmm2        # 8a <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x8a>
  8a:	mulss  %xmm1,%xmm2
  8e:	addss  0x0(%rip),%xmm2        # 96 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x96>
  96:	shr    $0x17,%eax
  99:	mulss  %xmm1,%xmm2
  9d:	addss  0x0(%rip),%xmm2        # a5 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0xa5>
  a5:	add    $0xffffff81,%eax
  a8:	cvtsi2ss %eax,%xmm3
  ac:	mulss  %xmm0,%xmm1
  b0:	mulss  %xmm2,%xmm1
  b4:	mulss  0x0(%rip),%xmm0        # bc <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0xbc>
  bc:	addss  %xmm3,%xmm0
  c0:	addss  %xmm1,%xmm0
  c4:	mulss  %xmm6,%xmm0
  c8:	movss  0x0(%rip),%xmm1        # d0 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0xd0>
  d0:	addss  %xmm0,%xmm1
  d4:	movd   %xmm1,%eax
  d8:	add    $0x20,%eax
  db:	mov    %eax,%ecx
  dd:	and    $0x3f,%ecx
  e0:	addss  0x0(%rip),%xmm1        # e8 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0xe8>
  e8:	mov    0x0(%rip),%r8        # ef <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0xef>
  ef:	movss  (%r8,%rcx,8),%xmm2
  f5:	subss  %xmm1,%xmm0
  f9:	movss  0x0(%rip),%xmm1        # 101 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x101>
 101:	mulss  %xmm0,%xmm1
 105:	addss  0x0(%rip),%xmm1        # 10d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x10d>
 10d:	mulss  %xmm0,%xmm1
 111:	and    $0xffffffc0,%eax
 114:	shl    $0x11,%eax
 117:	add    $0x3f800000,%eax
 11c:	movd   %eax,%xmm0
 120:	mulss  %xmm2,%xmm1
 124:	addss  %xmm2,%xmm1
 128:	mulss  %xmm0,%xmm1
 12c:	andps  0x0(%rip),%xmm1        # 133 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x133>
 133:	andps  0x0(%rip),%xmm5        # 13a <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x13a>
 13a:	orps   %xmm1,%xmm5
 13d:	movss  0x4(%rdx),%xmm7
 142:	xorps  %xmm0,%xmm0
 145:	ucomiss %xmm0,%xmm7
 148:	jne    160 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x160>
 14a:	jp     160 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x160>
 14c:	ucomiss %xmm0,%xmm6
 14f:	jne    26d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x26d>
 155:	jp     26d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x26d>
 15b:	jmp    3e0 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3e0>
 160:	movd   %xmm7,%ecx
 164:	and    $0x7fffffff,%ecx
 16a:	mov    %ecx,%eax
 16c:	sub    0x4(%rsi),%eax
 16f:	cmp    0x8(%rsi),%eax
 172:	ja     3e0 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3e0>
 178:	add    $0x4afb0d,%ecx
 17e:	mov    %ecx,%eax
 180:	and    $0x7fffff,%ecx
 186:	add    $0x3f3504f3,%ecx
 18c:	movd   %ecx,%xmm1
 190:	movss  0x0(%rip),%xmm0        # 198 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x198>
 198:	addss  %xmm1,%xmm0
 19c:	addss  0x0(%rip),%xmm1        # 1a4 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x1a4>
 1a4:	divss  %xmm1,%xmm0
 1a8:	movaps %xmm0,%xmm1
 1ab:	mulss  %xmm0,%xmm1
 1af:	movss  0x0(%rip),%xmm2        # 1b7 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x1b7>
 1b7:	mulss  %xmm1,%xmm2
 1bb:	addss  0x0(%rip),%xmm2        # 1c3 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x1c3>
 1c3:	shr    $0x17,%eax
 1c6:	mulss  %xmm1,%xmm2
 1ca:	addss  0x0(%rip),%xmm2        # 1d2 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x1d2>
 1d2:	add    $0xffffff81,%eax
 1d5:	xorps  %xmm3,%xmm3
 1d8:	cvtsi2ss %eax,%xmm3
 1dc:	mulss  %xmm0,%xmm1
 1e0:	mulss  %xmm2,%xmm1
 1e4:	mulss  0x0(%rip),%xmm0        # 1ec <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x1ec>
 1ec:	addss  %xmm3,%xmm0
 1f0:	addss  %xmm1,%xmm0
 1f4:	mulss  %xmm6,%xmm0
 1f8:	movss  0x0(%rip),%xmm1        # 200 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x200>
 200:	addss  %xmm0,%xmm1
 204:	movd   %xmm1,%eax
 208:	add    $0x20,%eax
 20b:	mov    %eax,%ecx
 20d:	and    $0x3f,%ecx
 210:	addss  0x0(%rip),%xmm1        # 218 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x218>
 218:	mov    0x0(%rip),%r8        # 21f <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x21f>
 21f:	movss  (%r8,%rcx,8),%xmm2
 225:	subss  %xmm1,%xmm0
 229:	movss  0x0(%rip),%xmm1        # 231 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x231>
 231:	mulss  %xmm0,%xmm1
 235:	addss  0x0(%rip),%xmm1        # 23d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x23d>
 23d:	mulss  %xmm0,%xmm1
 241:	and    $0xffffffc0,%eax
 244:	shl    $0x11,%eax
 247:	add    $0x3f800000,%eax
 24c:	movd   %eax,%xmm0
 250:	mulss  %xmm2,%xmm1
 254:	addss  %xmm2,%xmm1
 258:	mulss  %xmm0,%xmm1
 25c:	andps  0x0(%rip),%xmm1        # 263 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x263>
 263:	andps  0x0(%rip),%xmm7        # 26a <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x26a>
 26a:	orps   %xmm1,%xmm7
 26d:	movss  0x8(%rdx),%xmm1
 272:	xorps  %xmm0,%xmm0
 275:	ucomiss %xmm0,%xmm1
 278:	jne    290 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x290>
 27a:	jp     290 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x290>
 27c:	ucomiss %xmm0,%xmm6
 27f:	jne    399 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x399>
 285:	jp     399 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x399>
 28b:	jmp    419 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x419>
 290:	movd   %xmm1,%ecx
 294:	and    $0x7fffffff,%ecx
 29a:	mov    %ecx,%eax
 29c:	sub    0x4(%rsi),%eax
 29f:	cmp    0x8(%rsi),%eax
 2a2:	ja     419 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x419>
 2a8:	add    $0x4afb0d,%ecx
 2ae:	mov    %ecx,%eax
 2b0:	and    $0x7fffff,%ecx
 2b6:	add    $0x3f3504f3,%ecx
 2bc:	movd   %ecx,%xmm2
 2c0:	movss  0x0(%rip),%xmm0        # 2c8 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2c8>
 2c8:	addss  %xmm2,%xmm0
 2cc:	addss  0x0(%rip),%xmm2        # 2d4 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2d4>
 2d4:	divss  %xmm2,%xmm0
 2d8:	movaps %xmm0,%xmm2
 2db:	mulss  %xmm0,%xmm2
 2df:	movss  0x0(%rip),%xmm3        # 2e7 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2e7>
 2e7:	mulss  %xmm2,%xmm3
 2eb:	addss  0x0(%rip),%xmm3        # 2f3 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x2f3>
 2f3:	shr    $0x17,%eax
 2f6:	mulss  %xmm2,%xmm3
 2fa:	addss  0x0(%rip),%xmm3        # 302 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x302>
 302:	add    $0xffffff81,%eax
 305:	cvtsi2ss %eax,%xmm4
 309:	mulss  %xmm0,%xmm2
 30d:	mulss  %xmm3,%xmm2
 311:	mulss  0x0(%rip),%xmm0        # 319 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x319>
 319:	addss  %xmm4,%xmm0
 31d:	addss  %xmm2,%xmm0
 321:	mulss  %xmm0,%xmm6
 325:	movss  0x0(%rip),%xmm0        # 32d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x32d>
 32d:	addss  %xmm6,%xmm0
 331:	movd   %xmm0,%eax
 335:	add    $0x20,%eax
 338:	mov    %eax,%ecx
 33a:	and    $0x3f,%ecx
 33d:	addss  0x0(%rip),%xmm0        # 345 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x345>
 345:	mov    0x0(%rip),%rdx        # 34c <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x34c>
 34c:	movss  (%rdx,%rcx,8),%xmm2
 351:	subss  %xmm0,%xmm6
 355:	movss  0x0(%rip),%xmm0        # 35d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x35d>
 35d:	mulss  %xmm6,%xmm0
 361:	addss  0x0(%rip),%xmm0        # 369 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x369>
 369:	mulss  %xmm6,%xmm0
 36d:	and    $0xffffffc0,%eax
 370:	shl    $0x11,%eax
 373:	add    $0x3f800000,%eax
 378:	movd   %eax,%xmm3
 37c:	mulss  %xmm2,%xmm0
 380:	addss  %xmm2,%xmm0
 384:	mulss  %xmm3,%xmm0
 388:	andps  0x0(%rip),%xmm0        # 38f <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x38f>
 38f:	andps  0x0(%rip),%xmm1        # 396 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x396>
 396:	orps   %xmm0,%xmm1
 399:	movss  %xmm5,(%rdi)
 39d:	movss  %xmm7,0x4(%rdi)
 3a2:	movss  %xmm1,0x8(%rdi)
 3a7:	add    $0x20,%rsp
 3ab:	pop    %rbx
 3ac:	pop    %r14
 3ae:	pop    %r15
 3b0:	ret
 3b1:	movaps %xmm6,%xmm0
 3b4:	movaps %xmm5,%xmm1
 3b7:	mov    %rdi,%rbx
 3ba:	mov    %rsi,%r15
 3bd:	mov    %rdx,%r14
 3c0:	movss  %xmm6,(%rsp)
 3c5:	call   3ca <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3ca>
 3ca:	movss  (%rsp),%xmm6
 3cf:	mov    %r14,%rdx
 3d2:	mov    %r15,%rsi
 3d5:	mov    %rbx,%rdi
 3d8:	movaps %xmm0,%xmm5
 3db:	jmp    13d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x13d>
 3e0:	movaps %xmm6,%xmm0
 3e3:	movaps %xmm7,%xmm1
 3e6:	mov    %rdi,%rbx
 3e9:	mov    %rsi,%r15
 3ec:	mov    %rdx,%r14
 3ef:	movaps %xmm5,0x10(%rsp)
 3f4:	movss  %xmm6,(%rsp)
 3f9:	call   3fe <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x3fe>
 3fe:	movss  (%rsp),%xmm6
 403:	movaps 0x10(%rsp),%xmm5
 408:	mov    %r14,%rdx
 40b:	mov    %r15,%rsi
 40e:	mov    %rbx,%rdi
 411:	movaps %xmm0,%xmm7
 414:	jmp    26d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x26d>
 419:	movaps %xmm6,%xmm0
 41c:	mov    %rdi,%rbx
 41f:	movaps %xmm5,0x10(%rsp)
 424:	movaps %xmm7,(%rsp)
 428:	call   42d <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x42d>
 42d:	movaps (%rsp),%xmm7
 431:	movaps 0x10(%rsp),%xmm5
 436:	mov    %rbx,%rdi
 439:	movaps %xmm0,%xmm1
 43c:	jmp    399 <<moxcms::trc::ToneCurveEvaluatorPureGamma>::fallback_rgb+0x399>

Disassembly of section .text._RNvMs5_NtCs8N3SupDdS21_6moxcms3trcNtB5_27ToneCurveEvaluatorPureGamma3new:
