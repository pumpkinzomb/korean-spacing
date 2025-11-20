import KoreanSpacing from './index';

async function main() {
  const spacing = new KoreanSpacing();

  try {
    await spacing.load();

    const testTexts = [
      '안녕하세요반갑습니다',
      '오늘날씨가좋네요',
      '한국어띄어쓰기는어렵습니다',
      '머신러닝모델을만들고있습니다',
      '여기에줄바꿈이필요합니다',
    ];

    console.log('=== 단일 텍스트 교정 ===');
    for (const text of testTexts) {
      const result = await spacing.correct(text);
      console.log(`입력: ${text}`);
      console.log(`출력: ${JSON.stringify(result)}`);
      console.log('');
    }

    console.log('=== 배치 처리 ===');
    const results = await spacing.correctBatch(testTexts);
    testTexts.forEach((text, idx) => {
      console.log(`${text} -> ${JSON.stringify(results[idx])}`);
    });
  } catch (error) {
    console.error('오류 발생:', error);
    console.log('모델 파일이 없습니다. 먼저 학습 스크립트를 실행해주세요.');
  }
}

main();

