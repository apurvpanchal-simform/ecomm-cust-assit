# FAQ: Payments
**Category:** Payment | **Version:** 1.0 | **Last Updated:** June 2026
**RAG Tags:** payment, billing, checkout, credit card, debit card, UPI, net banking, EMI, wallet, refund, invoice, coupon, discount, promo code, gift card, failed transaction, payment security, COD, cash on delivery

---

## Overview

This document covers all frequently asked questions related to payments on our ecommerce platform. Topics include supported payment methods, checkout process, failed transactions, EMI options, coupons and gift cards, billing, and payment security. This file is optimized for Retrieval-Augmented Generation (RAG) and each section is self-contained to support accurate, context-rich responses.

---

## 1. Accepted Payment Methods

---

### Q1.1: What payment methods do you accept?

**Answer:**
We accept a wide range of payment methods to ensure a convenient checkout experience:

**Cards:**
- Visa, Mastercard, American Express, Rupay (credit and debit cards)
- Prepaid cards

**UPI (Unified Payments Interface):**
- Google Pay (GPay)
- PhonePe
- Paytm UPI
- BHIM UPI
- Any bank UPI handle (e.g., @ybl, @okaxis, @upi)

**Net Banking:**
- All major Indian banks including HDFC, ICICI, SBI, Axis, Kotak, and 50+ others

**Digital Wallets:**
- Paytm Wallet
- Mobikwik
- Amazon Pay
- Freecharge

**Buy Now, Pay Later (BNPL):**
- Simpl
- LazyPay
- ZestMoney

**EMI (Equated Monthly Installments):**
- Credit card EMI (3, 6, 9, 12-month plans)
- Cardless EMI via select bank partners

**Cash on Delivery (COD):**
- Available for eligible orders and pin codes (see Q1.4)

**Store Credit / Gift Cards:**
- Can be applied at checkout (see Section 6)

---

### Q1.2: Can I use multiple payment methods for a single order?

**Answer:**
Yes, in certain combinations. You can combine:
- **Store credit or gift card balance** with any other payment method (card, UPI, wallet, etc.)
- **Coupon discount** with any payment method

However, you cannot split a payment between two credit/debit cards, or between two UPI accounts. The remaining balance after applying store credit/gift cards must be paid using a single payment method.

---

### Q1.3: Do you accept international credit/debit cards?

**Answer:**
Yes. We accept internationally issued Visa, Mastercard, and American Express cards. Please note:

- International card transactions may be subject to **foreign transaction fees** charged by your issuing bank — we do not control or charge these.
- **Currency conversion** is handled by your bank; all prices on our site are displayed in INR (Indian Rupees).
- Some international cards may require **3D Secure (OTP) authentication**, which your bank will facilitate.
- If your international card is declined, try using PayPal (if available) or contact your bank to authorize international online transactions.

---

### Q1.4: Is Cash on Delivery (COD) available?

**Answer:**
Cash on Delivery is available for eligible orders. COD availability depends on:

- **Pin code serviceability** — COD is available in most metros and Tier-1 cities; enter your pin code on the product or checkout page to check.
- **Order value** — COD is available for orders up to **₹10,000**. Orders above this threshold require prepaid payment.
- **Product type** — Some categories (e.g., large appliances, custom/personalized items) are prepaid-only.

**COD fee:** A handling fee of **₹40** is added to COD orders.

**At delivery:** Please keep the exact cash amount ready. Our delivery partners may not carry change.

---

### Q1.5: Can I pay using cryptocurrency?

**Answer:**
Currently, we do not accept cryptocurrency as a payment method. We accept only the methods listed in Q1.1. If you'd like to see crypto payments added, you can submit a feature request through our feedback portal at **ourstore.com/feedback**.

---

## 2. Checkout & Payment Process

---

### Q2.1: How does the checkout payment process work?

**Answer:**
Here's a step-by-step overview of our checkout payment process:

1. **Add items to your cart** and proceed to checkout.
2. **Select or enter your shipping address**.
3. **Choose a delivery option** (standard, express, etc.).
4. **Select your payment method** from the available options.
5. **Apply any coupons, gift cards, or store credit** (optional).
6. **Review your order summary** including item prices, shipping fees, taxes, and total.
7. **Confirm and place your order** — you may be redirected to your bank's page for authentication (OTP or 3D Secure).
8. Upon successful payment, you'll see an **Order Confirmation** page and receive a confirmation email.

The entire process is encrypted and secure. We use **PCI-DSS compliant** payment gateways.

---

### Q2.2: Is it safe to enter my card details on your website?

**Answer:**
Yes. Our payment process uses industry-standard security measures:

- **SSL/TLS Encryption** — All data transmitted between your browser and our servers is encrypted (look for the padlock icon and "https://" in your browser).
- **PCI-DSS Compliance** — We are fully compliant with the Payment Card Industry Data Security Standard.
- **Tokenization** — Your card details are tokenized (converted to a non-sensitive token) by our payment gateway. We never store your full card number on our servers.
- **3D Secure (OTP verification)** — An additional authentication layer for card transactions, supported by all major Indian banks.

We strongly advise against completing payments on public Wi-Fi. Always use a trusted, private network.

---

### Q2.3: Will I be charged immediately or when my order ships?

**Answer:**
For most orders, your payment is **captured (charged) when you place the order**, not when it ships. This is standard for prepaid orders.

**Exception:** For pre-order or back-order items, we may authorize the amount at order placement but only capture (charge) the payment when the item is ready to ship. You'll be informed of this at checkout.

For **COD orders**, payment is collected at the time of delivery.

---

### Q2.4: Can I change my payment method after placing an order?

**Answer:**
Once an order is placed and payment is processed, you **cannot change the payment method**. However:

- If your order **hasn't shipped yet**, you may be able to cancel it and re-place it with a different payment method (see Returns & Cancellation FAQ).
- If a payment **failed** but you received an order confirmation, contact support — we'll help you complete the payment with your preferred method.

---

### Q2.5: Why am I being asked for an OTP during payment?

**Answer:**
OTP (One-Time Password) authentication is part of **3D Secure verification**, a security standard used by banks to confirm that you are the legitimate cardholder making the transaction. This is mandatory for most card and UPI transactions in India under RBI guidelines.

The OTP is sent by **your bank** (not by us) to your registered mobile number. It expires within **3–5 minutes**.

If you don't receive the OTP:
- Ensure your mobile number is registered with your bank.
- Check for network signal issues.
- Request a new OTP after the timer expires.
- Contact your bank if the problem persists.

---

## 3. Failed & Declined Transactions

---

### Q3.1: My payment failed. Why did this happen?

**Answer:**
Payments can fail for several reasons:

**Card-related:**
- Incorrect card number, expiry date, or CVV
- Card expired
- Insufficient funds or credit limit exceeded
- Card not enabled for online/international transactions
- Bank declined for fraud prevention

**UPI-related:**
- Incorrect UPI PIN
- UPI app server downtime
- Daily transaction limit exceeded on your UPI handle

**Net Banking:**
- Incorrect credentials
- Bank server downtime or maintenance
- Session timeout

**General:**
- Internet connectivity issues during transaction
- Browser/app crash mid-payment

**What to do:** Check your bank statement to confirm whether the amount was deducted. If it was, see Q3.2. If not, try again with correct details or a different payment method.

---

### Q3.2: My payment failed but money was deducted from my account. What happens now?

**Answer:**
This situation occurs due to a network or timeout issue where your bank processed the deduction but our system didn't receive confirmation. Here's what happens:

1. **Within 24–48 hours**, the payment gateway performs an automatic reconciliation and either:
   - Confirms the payment, completing your order
   - Identifies it as a failed transaction and initiates a **refund**

2. If your order is not confirmed within 48 hours, the deducted amount is automatically **refunded to your original payment method** within 5–7 business days.

3. You can also contact our support team at **support@ourstore.com** with your order ID and the deduction details — we'll investigate and expedite the resolution.

**Do not attempt to pay again immediately** — if the original payment gets confirmed, you may be double-charged, which will then require a separate refund process.

---

### Q3.3: My card keeps getting declined despite having sufficient funds. What should I do?

**Answer:**
If your card is declining despite sufficient balance, try the following:

1. **Verify card details** — Re-enter your card number, expiry date, and CVV carefully.
2. **Enable online transactions** — Log in to your bank's app or portal and ensure your card is enabled for online/e-commerce transactions.
3. **Check transaction limits** — Your card may have a daily online spending limit. Check and increase it via your bank's app.
4. **Try incognito/private browsing** — Browser extensions or cached data can sometimes interfere.
5. **Try a different browser or device**.
6. **Contact your bank** — They can see exactly why the transaction is being declined on their end.
7. **Use an alternative payment method** — UPI or net banking usually have fewer restrictions.

---

### Q3.4: My UPI payment keeps failing. What should I do?

**Answer:**
Common fixes for UPI payment failures:

1. **Check UPI PIN** — Ensure you're entering the correct 4 or 6-digit UPI PIN.
2. **UPI app update** — Make sure your GPay/PhonePe/Paytm app is updated to the latest version.
3. **Bank server status** — Sometimes your bank's UPI service is under maintenance; try after 30 minutes.
4. **Linked account balance** — Confirm that your bank account linked to UPI has sufficient balance.
5. **Daily limit** — NPCI sets a daily UPI transaction limit of ₹1,00,000. Check if you've exceeded it.
6. **Try a different UPI app** — If GPay fails, try PhonePe or BHIM.
7. **Re-link your bank account** — In your UPI app, delink and re-add your bank account.

---

## 4. EMI (Equated Monthly Installments)

---

### Q4.1: What EMI options are available?

**Answer:**
We offer EMI on credit cards and select debit cards from the following banks:

**Credit Card EMI:** Available on Visa, Mastercard, and Rupay credit cards from HDFC, ICICI, SBI, Axis, Kotak, Citibank, HSBC, and others.

**Tenures available:** 3, 6, 9, 12, 18, and 24 months (availability depends on bank and order amount).

**Cardless/No-Cost EMI:** Available via Bajaj Finserv, ZestMoney, and select bank partnerships for eligible customers.

**Minimum order value for EMI:** ₹3,000

At checkout, select **"EMI"** as your payment method and choose your bank and preferred tenure to see the applicable interest rate or no-cost EMI offers.

---

### Q4.2: What is No-Cost EMI? Is there really no extra charge?

**Answer:**
No-Cost EMI means you pay exactly the product price spread across installments — **no interest charged on top**. The interest amount is discounted from the product price upfront, so the total you pay equals the original price.

**How it works (example):**
- Product price: ₹12,000
- 3-month No-Cost EMI: ₹4,000/month × 3 = ₹12,000 total (no extra cost)

**Important notes:**
- Your credit card statement may show an interest charge from your bank, but this is offset by the upfront discount already applied.
- Processing fee: Some banks charge a one-time processing fee of ₹99–₹299, which is charged by your bank — not by us.
- No-Cost EMI availability varies by bank, card type, and product.

---

### Q4.3: How do I convert my credit card purchase to EMI?

**Answer:**
You can convert to EMI in two ways:

**At checkout:**
1. Select **"Credit Card"** as your payment method.
2. Enter your card details.
3. Before confirming, select the **"Convert to EMI"** option.
4. Choose your preferred tenure (3/6/9/12 months).
5. Confirm the order.

**Post-purchase (via your bank):**
If you missed the EMI option at checkout, you can contact your credit card issuer directly within **30 days** of the transaction to convert it to EMI. This is handled entirely by your bank.

---

### Q4.4: If I return an EMI purchase, how is the refund processed?

**Answer:**
If you return an item purchased on EMI:

- The **full product amount** (minus any applicable return fees) is refunded to your credit card.
- Your EMI will be **cancelled** once the refund is processed by your bank.
- Any interest already paid may or may not be refunded depending on your bank's policy — contact your bank directly for details.
- Refunds are typically processed within **5–7 business days** after the return is confirmed.

---

## 5. Coupons, Promo Codes & Discounts

---

### Q5.1: How do I apply a coupon or promo code?

**Answer:**
To apply a coupon code at checkout:

1. Add items to your cart and proceed to checkout.
2. On the **Order Summary** page, look for the **"Apply Coupon / Promo Code"** field.
3. Enter your code exactly as provided (codes are case-sensitive).
4. Click **"Apply"**.
5. The discount will be reflected in your order total before payment.

**Note:** Only one coupon code can be applied per order, unless stated otherwise in the offer terms.

---

### Q5.2: My coupon code isn't working. Why?

**Answer:**
Common reasons a coupon may not work:

- **Expired** — The code has passed its validity date. Check the offer's expiry date.
- **Minimum order value not met** — Many coupons require a minimum cart value (e.g., "₹200 off on orders above ₹1,500").
- **Category/product restriction** — The coupon may be valid only on specific brands, categories, or items.
- **Already used** — The code may be single-use and was already redeemed on your account.
- **Case sensitivity** — Enter the code exactly as shown, including uppercase/lowercase.
- **Account restriction** — Some coupons are for first-time users, new accounts, or specific customer segments.

If your code should be valid and still isn't working, contact support with the coupon code and we'll look into it.

---

### Q5.3: Can I use more than one coupon on the same order?

**Answer:**
In most cases, only **one coupon code** can be applied per order. However, you can combine a coupon with:
- **Store credit**
- **Gift card balance**
- **Loyalty reward points**
- **Bank offers** (e.g., 10% cashback on HDFC cards — these are applied by your bank, not by us)

Combining two coupon codes in the same order is not supported unless explicitly stated in the promotion terms.

---

### Q5.4: Where can I find active coupon codes and offers?

**Answer:**
You can find our current offers and coupons:
- **Homepage banner** — featured deals and flash sales
- **Offers page** — ourstore.com/offers
- **Email newsletter** — subscribe to receive exclusive codes
- **SMS/WhatsApp** — opt in to promotional messages
- **App notifications** — enable notifications on our mobile app
- **Social media** — follow us on Instagram, Facebook, and Twitter for exclusive codes

We never charge for coupon codes. If you encounter a third-party site asking you to pay for codes claiming to be from us, it is not affiliated with us.

---

## 6. Gift Cards & Store Credit

---

### Q6.1: How do I purchase a gift card?

**Answer:**
Gift cards are available in the following denominations: **₹250, ₹500, ₹1,000, ₹2,000, ₹5,000, ₹10,000**.

To purchase:
1. Go to **ourstore.com/gift-cards**.
2. Choose the denomination and format (digital or physical).
3. Add a personalized message (optional).
4. Proceed to checkout — gift cards can be purchased using any payment method.

**Digital gift cards** are delivered to the recipient's email within 1 hour of purchase.
**Physical gift cards** are shipped within 3–5 business days.

---

### Q6.2: How do I redeem a gift card?

**Answer:**
To redeem a gift card at checkout:

1. Add items to your cart and go to checkout.
2. In the **"Apply Gift Card"** field, enter the 16-digit gift card code.
3. Click **"Apply"**.
4. The gift card balance will be deducted from your order total.
5. If the order value exceeds the gift card balance, you can pay the remaining amount with any other payment method.

**Gift card terms:**
- Valid for **2 years** from date of issue
- Non-transferable and cannot be exchanged for cash
- Can be used for multiple purchases until the balance is exhausted

---

### Q6.3: What is store credit and how do I use it?

**Answer:**
Store credit is a balance in your account that can be used as payment for future orders. You may receive store credit from:
- **Refunds** (when original payment method is not eligible for direct refund)
- **Loyalty reward conversions**
- **Compensation for order issues**
- **Promotional campaigns**

To use store credit:
1. At checkout, look for the **"Use Store Credit"** toggle.
2. Enable it — your available store credit balance will be shown.
3. The credit will be applied automatically up to your order total.
4. Any remaining balance stays in your account for future use.

---

## 7. Invoices & Billing

---

### Q7.1: Where can I find my invoice?

**Answer:**
Invoices are available in your account:

1. Go to **My Account → Orders**.
2. Click on the relevant order.
3. Click **"Download Invoice"** (PDF format).

Invoices are generated **after your order is confirmed and payment is successful**. For COD orders, the invoice is available after delivery.

---

### Q7.2: Can I get a GST invoice for my order?

**Answer:**
Yes. To receive a GST (Goods and Services Tax) invoice:

1. Ensure your **GSTIN is saved** in Account Settings → Billing Information.
2. Enter your GSTIN during checkout in the billing details section.
3. The invoice will include your GSTIN, HSN codes, and applicable GST breakdown.

For already-placed orders, contact support with your order number and GSTIN and we'll issue a revised invoice within **2 business days**.

---

### Q7.3: My invoice shows an incorrect amount. What should I do?

**Answer:**
If you notice a discrepancy in your invoice amount:

1. Compare the invoice total with your bank statement/UPI debit.
2. Check if any discount, coupon, or store credit was applied — the invoice reflects the final charged amount.
3. If there's still a mismatch, contact support at **support@ourstore.com** with:
   - Your order number
   - The invoice amount shown
   - The amount charged to your account (screenshot/statement)
4. We'll investigate and issue a corrected invoice or process a differential refund within **3–5 business days**.

---

## 8. Refunds from Payments

---

### Q8.1: How long does a payment refund take?

**Answer:**
Refund timelines depend on the original payment method:

| Payment Method | Refund Timeline |
|---|---|
| Credit Card | 5–7 business days |
| Debit Card | 5–7 business days |
| UPI | 2–5 business days |
| Net Banking | 3–7 business days |
| Paytm/Wallet | 1–3 business days |
| COD (Store Credit) | Immediate |
| COD (Bank Transfer) | 5–7 business days |
| Gift Card | Same day (balance restored) |

**Note:** Refunds are initiated by us immediately upon return/cancellation confirmation. The timeline above reflects your bank/payment provider's processing time, which we do not control.

---

### Q8.2: I haven't received my refund after the stated timeline. What should I do?

**Answer:**
If your refund is overdue:

1. Check your **bank statement, UPI history, or wallet balance** — it may already be credited without a notification.
2. Go to **My Account → Orders → [Order] → Refund Status** to verify if the refund was initiated from our end.
3. If initiated and still not received, contact support with:
   - Order number
   - Refund reference number (available in your account)
   - Screenshot of your payment account showing no credit
4. We'll raise a formal dispute with the payment gateway within **24 hours**.

---

## 9. Payment Security & Fraud Prevention

---

### Q9.1: What should I do if I see an unauthorized charge from your store?

**Answer:**
If you see a charge you don't recognize:

1. **Check your order history** — it may be a legitimate order you forgot about.
2. **Check if a family member** used your saved payment method or account.
3. If the charge is truly unauthorized:
   - **Change your account password immediately** (see Account FAQ).
   - **Contact your bank** to dispute the charge and block your card.
   - **Contact us** at security@ourstore.com with transaction details — we'll investigate and freeze any suspicious orders.

We will **never charge you** without a corresponding order in your account.

---

### Q9.2: Do you store my credit/debit card details?

**Answer:**
We do **not store your full card number** on our servers. When you save a card to your account:
- Only a **token** (a unique non-sensitive identifier generated by our payment gateway) is saved.
- This token is specific to our platform and cannot be used elsewhere.
- Your actual card number, CVV, and expiry date are handled entirely by our PCI-DSS certified payment gateway (Razorpay/Stripe/PayU).

For extra peace of mind, you can choose not to save your card and enter it fresh each time you checkout.

---

*End of FAQ: Payments*
*For payment-specific support, email payments@ourstore.com or chat with us live (Mon–Sat, 9am–8pm IST).*
