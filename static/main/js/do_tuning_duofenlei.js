function updateValue(val, spanId) {
  document.getElementById(spanId).innerHTML = val;
}

document.getElementById('submitButton').addEventListener('click', function() {
    var Loading = document.getElementById('loading');
    // 按钮置灰
    this.style.backgroundColor = '#444'; // 深色按钮背景
    this.disabled = true; // 禁用按钮，使其不可点击
    this.textContent = '优化模型中...'; // 更改按钮文字
    Loading.style.display = 'block';
    Loading.classList.add('spinner-grow'); // 添加旋转动画
    startTraining();
});

function startTraining() {
    const form = document.getElementById('tuningForm');
    const formData = new FormData(form);
    var csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;
    const data = {};
    var submitButton = document.getElementById('submitButton');
    var Loading = document.getElementById('loading');

    // 获取原始值
    var accuracy = document.getElementById('accuracy');
    var loss = document.getElementById('loss');
    var test_accuracy = document.getElementById('test-accuracy');

    // 获取标签
    var i_accuracy_down = document.getElementById('i-accuracy-down');
    var i_accuracy_up = document.getElementById('i-accuracy-up');
    var i_loss_down = document.getElementById('i-loss-down');
    var i_loss_up = document.getElementById('i-loss-up');
    var i_test_accuracy_down = document.getElementById('i-test-accuracy-down');
    var i_test_accuracy_up = document.getElementById('i-test-accuracy-up');


    formData.forEach((value, key) => {
        data[key] = value;
    });

    fetch('/tuning_lstm_duofenlei/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // 恢复点击按钮
            submitButton.style.backgroundColor = '';
            submitButton.disabled = false;
            submitButton.textContent = '确认并开始训练';
            console.log('该模型的准确率为：', data.accuracy);
            Loading.style.display = 'none';
            // 根据上次结果显示箭头的方向
            if (data.accuracy > accuracy.innerText.replace('%', '')){
                i_accuracy_down.style.display = 'none';
                i_accuracy_up.style.display = 'block';
            }else {
                i_accuracy_up.style.display = 'none';
                i_accuracy_down.style.display = 'block';
            }

            if (data.loss > loss.innerText.replace('%', '')){
                i_loss_down.style.display = 'none';
                i_loss_up.style.display = 'block';
            }else {
                i_loss_up.style.display = 'none';
                i_loss_down.style.display = 'block';
            }

            if (data.test_accuracy > test_accuracy.innerText.replace('%', '')){
                i_test_accuracy_down.style.display = 'none';
                i_test_accuracy_up.style.display = 'block';
            }else {
                i_test_accuracy_up.style.display = 'none';
                i_test_accuracy_down.style.display = 'block';
            }
            // 更新性能指标
            console.log(data.accuracy, data.test_accuracy, data.loss);
            document.getElementById('accuracy').style.width = data.accuracy + '%';
            document.getElementById('accuracy').innerText = data.accuracy + '%';
            document.getElementById('loss').style.width = data.loss + '%';
            document.getElementById('loss').innerText = data.loss + '%';
            document.getElementById('test-accuracy').style.width = data.test_accuracy + '%';
            document.getElementById('test-accuracy').innerText = data.test_accuracy + '%';
        } else {
            alert('训练失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



