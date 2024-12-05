app = angular.module('chatApp', [])
app.config(function($interpolateProvider) {
  $interpolateProvider.startSymbol('[[');
  $interpolateProvider.endSymbol(']]');
});
app.directive('scrollToBottom', function($timeout) {
    return {
      scope: {
        scrollToBottom: '='
      },
      link: function(scope, element) {
        scope.$watchCollection('scrollToBottom', function(newVal) {
          $timeout(function() {
            element[0].scrollTop = element[0].scrollHeight;
          }, 0);
        });
      }
    };
  });
app.controller('ChatController', function($scope, $timeout, $http) {
      
    $scope.messages = [
        { sender: 'bot', text: "Hi! I'm FinanceBot, your Query ChatBot ❤️" }
    ];
    $scope.userInput = '';
    $scope.isFullscreen = false;
    // Toggle chatbot visibility
    $scope.toggleChat = function() {
        const chatPopup = document.getElementById('chatPopup');
        chatPopup.style.display = chatPopup.style.display === 'block' ? 'none' : 'block';
    };
        // Toggle full-screen mode
        $scope.maximizeChat = function() {
          $scope.isFullscreen = !$scope.isFullscreen;
      };

// Send user message and get bot response
$scope.sendMessage = function() {
  if ($scope.userInput.trim() === '') return;

  // Push user's message to chat
  $scope.messages.push({ sender: 'user', text: $scope.userInput });

  const userMessage = $scope.userInput;
  $scope.userInput = ''; // Clear input field

  // Make GET request to Flask backend to get bot's response
  $http.get('/get', { params: { msg: userMessage } })
      .then(function(response) {
          const botReply = response.data.bot_reply;
          const graphUrl = response.data.graph_url || null;

         // If there's a graph, push it to chat
          if (graphUrl) {
              $scope.messages.push({ sender: 'bot', text: botReply, graphUrl: graphUrl });
          }
          else{
                // Push bot's text response
                $scope.messages.push({ sender: 'bot', text: botReply });
          }

          // Scroll chat body to the bottom
          const chatBody = document.getElementById('chatBody');
          chatBody.scrollTop = chatBody.scrollHeight;
      }, function(error) {
          console.error('Error fetching response from backend:', error);
      });
};


    // Handle "Enter" key press to send message
    $scope.handleKeyPress = function(event) {
        if (event.which === 13) {
            $scope.sendMessage();
        }
    };
});