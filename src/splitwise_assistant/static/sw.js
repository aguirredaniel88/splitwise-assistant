// Service Worker for PWA support
const CACHE_NAME = 'splitwise-assistant-v1';

self.addEventListener('install', (event) => {
  console.log('Service Worker installing...');
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  console.log('Service Worker activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
  return self.clients.claim();
});

// Minimal fetch handler - let network requests pass through
self.addEventListener('fetch', (event) => {
  // Don't cache - always go to network for fresh data
  event.respondWith(fetch(event.request));
});
